/*****************************************************************************
 ** ANGRYBIRDS AI AGENT FRAMEWORK
 ** Copyright (c) 2014, XiaoYu (Gary) Ge, Stephen Gould, Jochen Renz
 **  Sahan Abeyasinghe,Jim Keys,  Andrew Wang, Peng Zhang
 ** All rights reserved.
**This work is licensed under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
**To view a copy of this license, visit http://www.gnu.org/licenses/
 *****************************************************************************/
package ab.demo;

import java.awt.Point;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Collections;
import java.util.Comparator;

import javax.imageio.ImageIO;

// 2017-05-07 : ymkim1019
import java.net.*;
import java.io.*;
import java.lang.String;

import ab.demo.other.ActionRobot;
import ab.demo.other.Shot;
import ab.planner.TrajectoryPlanner;
import ab.utils.StateUtil;
import ab.vision.ABObject;
import ab.vision.ABType;
import ab.vision.GameStateExtractor.GameState;
import ab.vision.Vision;
// 2017-06-04 : ymkim1019
import ab.vision.VisionMBR;

// 2017-05-17 : jyham
import ab.vision.Preprocessor;

public class NaiveAgentDQN implements Runnable {

	private ActionRobot aRobot;
	private Random randomGenerator;
	public int currentLevel = 1;
	public int startLevel = 1;
	public int endLevel = 21;
	// 2017-05-07 : ymkim1019
	public String agent_ip = "127.0.0.1";
	public int agent_port = 2004;
	public Socket so;
	public DataInputStream in;
	public DataOutputStream out;
	public int stars = 0;
	
	public static int time_limit = 12;
	private Map<Integer,Integer> scores = new LinkedHashMap<Integer,Integer>();
	private Map<Integer,Integer> levelStars = new LinkedHashMap<Integer,Integer>();
	TrajectoryPlanner tp;
	private boolean firstShot;
	private Point prevTarget;
	
	
	public enum EnvToAgentJobId {
		OBSERVE;
	}
	
	public enum AgentToEnvJobId {
		ACT;
	}
	
	// a standalone implementation of the Naive Agent
	public NaiveAgentDQN() {
		
		aRobot = new ActionRobot();
		tp = new TrajectoryPlanner();
		prevTarget = null;
		firstShot = true;
		randomGenerator = new Random();
		// --- go to the Poached Eggs episode level selection page ---
		ActionRobot.GoFromMainMenuToLevelSelection();

	}
	
	// run the client
	public void run() {
		// 2017-04-01 : ymkim1019
		try {
			System.out.format("Connecting to the Agent..%s:%d\n", agent_ip, agent_port);
			so = new Socket(agent_ip, agent_port);
			System.out.format("Connected..\n");
			in = new DataInputStream(so.getInputStream());
			out = new DataOutputStream(so.getOutputStream());
		} catch (Exception e) {
			System.out.println("Fail to connect to the Agent..");
			return;
		}
		
		aRobot.loadLevel(currentLevel);
		while (true) {
			GameState state;
			try {
				state = solve();
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
				break;
			}
			// 2017-04-01 : ymkim1019
			// The shot has already been executed..
			if (state == GameState.WON) {
				try {
					Thread.sleep(3000);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				// 2017-04-01 : ymkim1019
				// Update the current stage score and stars
				int score = StateUtil.getScore(ActionRobot.proxy);
				stars = StateUtil.getStars(ActionRobot.proxy);

				if(!scores.containsKey(currentLevel))
					scores.put(currentLevel, score);
				else
				{
					if(scores.get(currentLevel) < score)
						scores.put(currentLevel, score);
				}
				
				if(!levelStars.containsKey(currentLevel))
					levelStars.put(currentLevel, stars);
				else
				{
					if(levelStars.get(currentLevel) < stars)
						levelStars.put(currentLevel, stars);
				}
				
				int totalScore = 0;
				for(Integer key: scores.keySet()){

					totalScore += scores.get(key);
					System.out.println(" Level " + key
							+ " Score: " + scores.get(key) + " Stars: " + levelStars.get(key));
				}
				System.out.println("Total Score: " + totalScore);
				
				currentLevel++;
				if (currentLevel > endLevel)
					currentLevel = startLevel;
				
				aRobot.loadLevel(currentLevel);
				// make a new trajectory planner whenever a new level is entered
				tp = new TrajectoryPlanner();

				// first shot on this level, try high shot first
				firstShot = true;
			} else if (state == GameState.LOST) {
				System.out.println("===========Restart===========");
				int totalScore = 0;
				for(Integer key: scores.keySet()){

					totalScore += scores.get(key);
					System.out.println(" Level " + key
							+ " Score: " + scores.get(key) + " Stars: " + levelStars.get(key));
				}
				System.out.println("Total Score: " + totalScore);
				aRobot.restartLevel();
			} else if (state == GameState.LEVEL_SELECTION) {
				System.out
				.println("Unexpected level selection page, go to the last current level : "
						+ currentLevel);
				aRobot.loadLevel(currentLevel);
			} else if (state == GameState.MAIN_MENU) {
				System.out
				.println("Unexpected main menu page, go to the last current level : "
						+ currentLevel);
				ActionRobot.GoFromMainMenuToLevelSelection();
				aRobot.loadLevel(currentLevel);
			} else if (state == GameState.EPISODE_MENU) {
				System.out
				.println("Unexpected episode menu page, go to the last current level : "
						+ currentLevel);
				ActionRobot.GoFromMainMenuToLevelSelection();
				aRobot.loadLevel(currentLevel);
			}

		}

	}

	private double distance(Point p1, Point p2) {
		return Math
				.sqrt((double) ((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y)
						* (p1.y - p2.y)));
	}
	
	public void makeActions(List<Integer> typeList, List<Integer> xList, List<Integer> yList, List<Double> angleList
			, List<Rectangle> targetList, Rectangle sling, int type)
	{
		for (int i=0;i<targetList.size();i++)
		{
			// estimate the trajectory
			Point targetPoint = new Point((int)targetList.get(i).getCenterX(), (int)targetList.get(i).getCenterY());
			ArrayList<Point> pts = tp.estimateLaunchPoint(sling, targetPoint);
			for (int j=0;j<pts.size();j++) // high and low shot
			{
				Point releasePoint = pts.get(j);
				double angle = tp.getReleaseAngle(sling, releasePoint);
				angle = Math.toDegrees(angle);
				typeList.add(type);
				xList.add((int)targetList.get(i).getCenterX());
				yList.add((int)targetList.get(i).getCenterY());
				angleList.add(angle);
			}
		}		
	}

	public void send_env_to_agent(Vision vision, int first_shot, int done, int n_stars, List<Rectangle> pigs
			, List<Rectangle> stones, List<Rectangle> woods, List<Rectangle> ices, List<Rectangle> tnts
			, ABType bird_type, Rectangle sling) throws IOException
	{
		System.out.println("send the environments to the agent..");
		
		// Img arr
		BufferedImage imgBuf = vision.getImageBuffer();
		// 2017-05-17 : jyham
		Preprocessor prep = new Preprocessor(imgBuf);
		imgBuf = prep.drawImage(imgBuf, false);
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ImageIO.write(imgBuf, "jpg", baos );
		baos.flush();
		byte[] imageInByte=baos.toByteArray();
		baos.close();
		int n_targets = pigs.size() + stones.size() + woods.size() + ices.size() + tnts.size();
		
		// candidate actions
		List<Integer> typeList = new ArrayList<Integer>();
		List<Integer> xList = new ArrayList<Integer>();
		List<Integer> yList = new ArrayList<Integer>();
		List<Double> angleList = new ArrayList<Double>();
		String[] typeStrings = {"pig", "stone", "wood", "ice", "tnt"}; 
		makeActions(typeList, xList, yList, angleList, pigs, sling, 0);
		makeActions(typeList, xList, yList, angleList, stones, sling, 1);
		makeActions(typeList, xList, yList, angleList, woods, sling, 2);
		makeActions(typeList, xList, yList, angleList, ices, sling, 3);
		makeActions(typeList, xList, yList, angleList, tnts, sling, 4);		
		
		int packetSize = 4 * 14 + typeList.size() * 20 + baos.size();
		System.out.println("packet size = " + packetSize);
		out.writeInt(packetSize);
		out.writeInt(EnvToAgentJobId.OBSERVE.ordinal()); // Job ID
		out.writeInt(first_shot); // first shot
		out.writeInt(done); // done
		out.writeInt(n_stars); // n_stars
		out.writeInt(pigs.size()); // # of pigs
		out.writeInt(stones.size()); // # of stones
		out.writeInt(woods.size()); // # of wood blocks
		out.writeInt(ices.size()); // # of ice blocks
		out.writeInt(tnts.size()); // # of TNTs
		out.writeInt(bird_type.id-4); // bird type on the sling
		out.writeInt(currentLevel); // level
		Point refPoint = tp.getReferencePoint(sling);
		out.writeInt((int)refPoint.getX());
		out.writeInt((int)refPoint.getY());
		out.writeInt(typeList.size()); // # of action candidates
		System.out.println("-------- observation --------");
		System.out.println("-> first shot : " + first_shot);
		System.out.println("-> done : " + done);
		System.out.println("-> n_stars : " + n_stars);
		System.out.println("-> # pigs : " + pigs.size());
		System.out.println("-> # stones : " + stones.size());
		System.out.println("-> # woods : " + woods.size());
		System.out.println("-> # ices : " + ices.size());
		System.out.println("-> # tnts : " + tnts.size());
		System.out.println("-> bird : " + bird_type.toString());
		System.out.format("-> sling ref point : (%d, %d)\n", (int)refPoint.getX(), (int)refPoint.getY());
		System.out.println("------- target list ----- : " + typeList.size());
		for (int i=0;i<typeList.size();i++)
		{
//			System.out.format("type=%s, (%d, %d), angle=%f\n"
//					, typeStrings[typeList.get(i)], xList.get(i)
//					, yList.get(i), angleList.get(i));
//			System.out.flush();			
			out.writeInt(typeList.get(i));
			out.writeInt(xList.get(i));
			out.writeInt(yList.get(i));
			out.writeDouble(angleList.get(i));
		}
		
		out.write(imageInByte);
		out.flush();
	}
	
	public GameState solve() throws IOException
	{

		// capture Image
		BufferedImage screenshot = ActionRobot.doScreenShot();
		
		// 2017-04-01 : ymkim1019
		//System.out.println("screen shot size = " + screenshot.getWidth() + "," + screenshot.getHeight());

		// process image
		Vision vision = new Vision(screenshot);
		VisionMBR vision_mbr = new VisionMBR(screenshot); // 2017-06-04 : ymkim1019
		
		// find the slingshot
		Rectangle sling = vision.findSlingshotMBR();

		// confirm the slingshot
		while (sling == null && aRobot.getState() == GameState.PLAYING) {
			System.out
			.println("No slingshot detected. Please remove pop up or zoom out");
			ActionRobot.fullyZoomOut();
			screenshot = ActionRobot.doScreenShot();
			vision = new Vision(screenshot);
			sling = vision.findSlingshotMBR();
		}
        // get all the pigs
 		List<Rectangle> pigs = vision_mbr.findPigsMBR();

		GameState state = aRobot.getState();

		// if there is a sling, then play, otherwise just skip.
		if (sling != null) {

			if (!pigs.isEmpty()) {
				// 2017-05-07 : ymkim1019
				List<Rectangle> stones = vision_mbr.findStonesMBR();
				List<Rectangle> woods = vision_mbr.findWoodMBR();
				List<Rectangle> ices = vision_mbr.findIceMBR();
				List<Rectangle> tnts = vision_mbr.findTNTsMBR();
								
				send_env_to_agent(vision, (firstShot)? 1 : 0, (firstShot)? 1 : 0, stars, pigs, stones
						, woods, ices, tnts, aRobot.getBirdTypeOnSling(), sling);
				
				int size = in.readInt();
				int job_id = in.readInt();
				int target_x = in.readInt();
				int target_y = in.readInt();
				double angle = in.readDouble();
				int tap_time = in.readInt();
				
				System.out.format("size=%d, job_id=%d, target=(%d, %d), angle=%f, tap_time=%d\n"
						, size, job_id, target_x, target_y, angle, tap_time);
				
				angle = Math.toRadians(angle);
				
				Point releasePoint = tp.findReleasePoint(sling, angle);
				Shot shot = new Shot();
			
				// Get the reference point
				Point refPoint = tp.getReferencePoint(sling);

				System.out.println("Release Point: " + releasePoint);
				System.out.println("Release Angle: " + Math.toDegrees(angle));
				
				int tapInterval = tap_time;
				Point _tpt = new Point(target_x, target_y);
				int tapTime = tp.getTapTime(sling, releasePoint, _tpt, tapInterval);
				int dx = (int)releasePoint.getX() - refPoint.x;
				int dy = (int)releasePoint.getY() - refPoint.y;
				shot = new Shot(refPoint.x, refPoint.y, dx, dy, 0, tapTime);

				// check whether the slingshot is changed. the change of the slingshot indicates a change in the scale.
				{
					ActionRobot.fullyZoomOut();
					screenshot = ActionRobot.doScreenShot();
					vision = new Vision(screenshot);
					Rectangle _sling = vision.findSlingshotMBR();
					if(_sling != null)
					{
						double scale_diff = Math.pow((sling.width - _sling.width),2) +  Math.pow((sling.height - _sling.height),2);
						if(scale_diff < 25)
						{
							if(dx < 0)
							{
								aRobot.cshoot(shot);
								state = aRobot.getState();
								if ( state == GameState.PLAYING )
								{
									screenshot = ActionRobot.doScreenShot();
									vision = new Vision(screenshot);
									List<Point> traj = vision.findTrajPoints();
									// 2017-04-01 : ymkim1019
									// below codes calibrate the trajectory module
									tp.adjustTrajectory(traj, sling, releasePoint);
									firstShot = false;
								}
							}
						}
						else
							System.out.println("Scale is changed, can not execute the shot, will re-segement the image");
					}
					else
						System.out.println("no sling detected, can not execute the shot, will re-segement the image");
				}
			}

		}
		return state;
	}

	public static void main(String args[]) {

		NaiveAgentDQN na = new NaiveAgentDQN();
		if (args.length > 0)
			na.currentLevel = Integer.parseInt(args[0]);
			
		na.run();

	}
}
