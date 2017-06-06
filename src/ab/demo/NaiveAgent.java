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

public class NaiveAgent implements Runnable {

	private ActionRobot aRobot;
	private Random randomGenerator;
	public int currentLevel = 1;
	// 2017-05-07 : ymkim1019
	public String agent_ip = "127.0.0.1";
	public int agent_port = 2004;
	public Socket so;
	public DataInputStream in;
	public DataOutputStream out;
	public int stars = 0;
	
	public static int time_limit = 12;
	private Map<Integer,Integer> scores = new LinkedHashMap<Integer,Integer>();
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
	public NaiveAgent() {
		
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
				int totalScore = 0;
				for(Integer key: scores.keySet()){

					totalScore += scores.get(key);
					System.out.println(" Level " + key
							+ " Score: " + scores.get(key) + " Stars: " + Integer.toString(stars));
				}
				System.out.println("Total Score: " + totalScore);
				aRobot.loadLevel(++currentLevel);
				// make a new trajectory planner whenever a new level is entered
				tp = new TrajectoryPlanner();

				// first shot on this level, try high shot first
				firstShot = true;
			} else if (state == GameState.LOST) {
				System.out.println("Restart");
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

	public void send_env_to_agent(Vision vision, int first_shot, int done, int n_stars, int n_pigs, int n_stones
			, int n_woods, int n_ices, int n_tnts, ABType bird_type) throws IOException
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
		
		out.writeInt(4 * 11 + baos.size());
		out.writeInt(EnvToAgentJobId.OBSERVE.ordinal()); // Job ID
		out.writeInt(first_shot); // first shot
		out.writeInt(done); // done
		out.writeInt(n_stars); // n_stars
		out.writeInt(n_pigs); // # of pigs
		out.writeInt(n_stones); // # of stones
		out.writeInt(n_woods); // # of wood blocks
		out.writeInt(n_ices); // # of ice blocks
		out.writeInt(n_tnts); // # of TNTs
		out.writeInt(bird_type.id-3); // bird type on the sling
		out.writeInt(currentLevel); // level
		out.write(imageInByte);
		out.flush();
		
		System.out.println("-------- observation --------");
		System.out.println("-> first shot : " + first_shot);
		System.out.println("-> done : " + done);
		System.out.println("-> n_stars : " + n_stars);
		System.out.println("-> # pigs : " + n_pigs);
		System.out.println("-> # stones : " + n_stones);
		System.out.println("-> # woods : " + n_woods);
		System.out.println("-> # ices : " + n_ices);
		System.out.println("-> # tnts : " + n_tnts);
		System.out.println("-> bird : " + bird_type.toString());
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
 		List<ABObject> pigs = vision.findPigsMBR();

		GameState state = aRobot.getState();

		// if there is a sling, then play, otherwise just skip.
		if (sling != null) {

			if (!pigs.isEmpty()) {
				// 2017-05-07 : ymkim1019
				List<Rectangle> stones = vision_mbr.findStonesMBR();
				List<Rectangle> woods = vision_mbr.findWoodMBR();
				List<Rectangle> ices = vision_mbr.findIceMBR();
				List<Rectangle> tnts = vision_mbr.findTNTsMBR();
								
				// sort
				Comparator<Rectangle> comp = new Comparator<Rectangle>() {
				      @Override
				      public int compare(final Rectangle object1, final Rectangle object2) {
				    	  if (object1.getX() == object2.getX())
				    	  { 	
				    		  return (int) (Math.round(object1.getY()) - Math.round(object2.getY()));
				    	  }
				    	  else
				    	  {
				    		  return (int) (Math.round(object1.getX()) - Math.round(object2.getX()));
				    	  }
				      }
				};
				Collections.sort(pigs, comp);
				Collections.sort(stones, comp);
				Collections.sort(woods, comp);
				Collections.sort(ices, comp);
				Collections.sort(tnts, comp);
				List<Rectangle> targets = new ArrayList<Rectangle>();
				targets.addAll(pigs);
				targets.addAll(stones);
				targets.addAll(woods);
				targets.addAll(ices);
				targets.addAll(tnts);
				
				send_env_to_agent(vision, (firstShot)? 1 : 0, (firstShot)? 1 : 0, stars, pigs.size(), stones.size()
						, woods.size(), ices.size(), tnts.size(), aRobot.getBirdTypeOnSling());
				
				int size = in.readInt();
				int job_id = in.readInt();
				int target = in.readInt();
				int high_shot = in.readInt();
				int tap_time = in.readInt();
				System.out.format("size=%d, job_id=%d, target=%d, high_shot=%d, tap_time=%d\n"
						, size, job_id, target, high_shot, tap_time);
				
				Point releasePoint = null;
				Shot shot = new Shot();
				int dx,dy;
				{
					/*
					// random pick up a pig
					ABObject pig = pigs.get(randomGenerator.nextInt(pigs.size()));
					
					Point _tpt = pig.getCenter();// if the target is very close to before, randomly choose a
					*/
					Rectangle targetObj = targets.get(target);
					Point _tpt = new Point();
					_tpt.setLocation(Math.round(targetObj.getX()+targetObj.getWidth()/2), Math.round(targetObj.getY()+targetObj.getHeight()/2));
					
					// point near it
					if (prevTarget != null && distance(prevTarget, _tpt) < 10) {
						double _angle = randomGenerator.nextDouble() * Math.PI * 2;
						_tpt.x = _tpt.x + (int) (Math.cos(_angle) * 10);
						_tpt.y = _tpt.y + (int) (Math.sin(_angle) * 10);
						System.out.println("Randomly changing to " + _tpt);
					}

					prevTarget = new Point(_tpt.x, _tpt.y);

					// estimate the trajectory
					ArrayList<Point> pts = tp.estimateLaunchPoint(sling, _tpt);
					
					// 2017-04-01 : ymkim1019
					System.out.println("# of launch points=" + pts.size());
					
					if (pts.size() == 1)
						releasePoint = pts.get(0);
					else if (pts.size() > 1)
					{
						if (high_shot == 1)
							releasePoint = pts.get(1);
						else
							releasePoint = pts.get(0);
					}
					else
					{
						System.out.println("No release point found for the target");
						System.out.println("Try a shot with 45 degree");
						releasePoint = tp.findReleasePoint(sling, Math.PI/4);
					}
					
					/*
					// do a high shot when entering a level to find an accurate velocity
					if (firstShot && pts.size() > 1) 
					{
						releasePoint = pts.get(1);
					}
					else if (pts.size() == 1)
						releasePoint = pts.get(0);
					else if (pts.size() == 2)
					{
						// randomly choose between the trajectories, with a 1 in
						// 6 chance of choosing the high one
						if (randomGenerator.nextInt(6) == 0)
							releasePoint = pts.get(1);
						else
							releasePoint = pts.get(0);
					}
					else
						if(pts.isEmpty())
						{
							System.out.println("No release point found for the target");
							System.out.println("Try a shot with 45 degree");
							releasePoint = tp.findReleasePoint(sling, Math.PI/4);
						}
					*/
					
					// Get the reference point
					Point refPoint = tp.getReferencePoint(sling);


					//Calculate the tapping time according the bird type 
					if (releasePoint != null) {
						double releaseAngle = tp.getReleaseAngle(sling,
								releasePoint);
						System.out.println("Release Point: " + releasePoint);
						System.out.println("Release Angle: "
								+ Math.toDegrees(releaseAngle));
						int tapInterval = 0;
						// 2017-06-06 ymkim1019
						tapInterval = tap_time;
						/*
						switch (aRobot.getBirdTypeOnSling()) 
						{

						case RedBird:
							tapInterval = 0; break;               // start of trajectory
						case YellowBird:
							tapInterval = 65 + randomGenerator.nextInt(25);break; // 65-90% of the way
						case WhiteBird:
							tapInterval =  70 + randomGenerator.nextInt(20);break; // 70-90% of the way
						case BlackBird:
							tapInterval =  70 + randomGenerator.nextInt(20);break; // 70-90% of the way
						case BlueBird:
							tapInterval =  65 + randomGenerator.nextInt(20);break; // 65-85% of the way
						default:
							tapInterval =  60;
						}
						*/

						int tapTime = tp.getTapTime(sling, releasePoint, _tpt, tapInterval);
						dx = (int)releasePoint.getX() - refPoint.x;
						dy = (int)releasePoint.getY() - refPoint.y;
						shot = new Shot(refPoint.x, refPoint.y, dx, dy, 0, tapTime);
					}
					else
						{
							System.err.println("No Release Point Found");
							return state;
						}
				}

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

		NaiveAgent na = new NaiveAgent();
		if (args.length > 0)
			na.currentLevel = Integer.parseInt(args[0]);
			
		na.run();

	}
}
