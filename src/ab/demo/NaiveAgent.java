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

// 2017-05-17 : jyham
import ab.vision.Preprocessor;
import java.awt.Color;

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
	
	public static int time_limit = 12;
	private Map<Integer,Integer> scores = new LinkedHashMap<Integer,Integer>();
	TrajectoryPlanner tp;
	private boolean firstShot;
	private Point prevTarget;
	
	private int prevStars;
	private boolean shouldWriteStars = false;
	private boolean prevFail = false;
	
	
	public enum EnvToAgentJobId {
		FROM_ENV_TO_AGENT_REQUEST_FOR_ACTION;
	}
	
	public enum AgentToEnvJobId {
		FROM_AGENT_TO_ENV_DO_ACTION;
	}
	
	// a standalone implementation of the Naive Agent
	public NaiveAgent() {
		
		aRobot = new ActionRobot();
		tp = new TrajectoryPlanner();
		prevTarget = null;
		firstShot = true;
		randomGenerator = new Random(); // should fix random seed ?
		// --- go to the Poached Eggs episode level selection page ---
		ActionRobot.GoFromMainMenuToLevelSelection();

	}
	
	// run the client
	public void run() {
		// 2017-04-01 : ymkim1019
		try {
			so = new Socket(agent_ip, agent_port);
			System.out.println("Connected to the Agent..");
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
				// 2017-06-07 jyham
				state = solve_angle();
				//state = solve();
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
				int stars = StateUtil.getStars(ActionRobot.proxy);
				prevStars = stars;
				shouldWriteStars = true;

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
				aRobot.loadLevel(++currentLevel); // TODO: modify here?
				// make a new trajectory planner whenever a new level is entered
				tp = new TrajectoryPlanner();

				// first shot on this level, try high shot first
				firstShot = true;
			} else if (state == GameState.LOST) {
				System.out.println("Restart");
				// jyham: to send failure
				prevFail = true;
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

	public BufferedImage send_env_to_agent(Vision vision) throws IOException
	{
		//System.out.println(aRobot.getState());
		
		BufferedImage imgBuf = vision.getImageBuffer();
		// 2017-05-17 : jyham
		Preprocessor prep = new Preprocessor(imgBuf);
		imgBuf = prep.drawImage(imgBuf, false);
		
		// 2017-06-03 : jyham
		ABType[] birds_seq = prep.getBirds();
		int max_birds_num = 10;
		
		// 2017-06-07 : jyham
		Rectangle sling = prep.getSling();
		int ground = prep.getGround();
		int pigs_num = prep.getPigsNum();
		
		System.out.println("send the environments to the agent..");
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ImageIO.write(imgBuf, "jpg", baos );
		baos.flush();
		byte[] imageInByte=baos.toByteArray();
		baos.close();
		//out.writeInt(4 + baos.size()); // Job ID + Img
		
		// job id, birds sequence, sling position (x, y, height), img 
		int data_size = 4 + 4*max_birds_num + 4*7 + baos.size();
		
		out.writeInt(data_size); // Job ID + birds sequence + Img
		out.writeInt(EnvToAgentJobId.FROM_ENV_TO_AGENT_REQUEST_FOR_ACTION.ordinal()); // Job ID
		
		if (birds_seq == null){
			for (int i=0; i<max_birds_num; i++){
				out.writeInt(0);
			}
		}
		//System.out.println(birds_seq.length);
		else{
			for (int i = 0; i < max_birds_num ; i++){
		
				if (i<birds_seq.length) {
					out.writeInt(birds_seq[i].id);
					System.out.println(birds_seq[i] + " " +birds_seq[i].id);
				}
				else out.writeInt(0);
			} // birds sequence
		}
		
		out.writeInt(sling.x);
		out.writeInt(sling.y);
		out.writeInt(sling.height);
		out.writeInt(ground);
		out.writeInt(pigs_num);
		
		if (shouldWriteStars){
			out.writeInt(prevStars);
			shouldWriteStars = false;
			prevStars = 0;
		}
		else if (prevFail){
			out.writeInt(-10);
			prevFail = false;
		}
		else{
			out.writeInt(0);
		}
		
		if(firstShot) out.writeInt(1);
		else out.writeInt(0);
		
		out.write(imageInByte);
		out.flush();
		
		return imgBuf;
	}
	
	
	public GameState solve_angle() throws IOException
	{
		// capture Image
		BufferedImage screenshot = ActionRobot.doScreenShot();
		int sc_w = (int) screenshot.getWidth();
		int sc_h = (int) screenshot.getHeight();
		System.out.println("screen shot size = " + sc_w + "," + sc_h);
		
		Vision vision = new Vision(screenshot);
		
		Rectangle sling = vision.findSlingshotMBR();
		
		while(sling == null && aRobot.getState() == GameState.PLAYING){
			System.out
			.println("No slingshot detected. Please remove pop up or zoom out");
			ActionRobot.fullyZoomOut();
			screenshot = ActionRobot.doScreenShot();
			vision = new Vision(screenshot);
			sling = vision.findSlingshotMBR();
		}
		
		List <ABObject> pigs = vision.findPigsMBR();
		
		GameState state = aRobot.getState();
		
		
		if (sling != null){
			if (!pigs.isEmpty()){
				
				BufferedImage img = send_env_to_agent(vision);
				
				int size = in.readInt();
				int job_id = in.readInt();
				int theta = in.readInt();
				System.out.format("size=%d, job_id=%d, data=%d\n", size, job_id, theta);
				
				double releaseAngle = Math.toRadians(theta);
				Point releasePoint = null;
				Shot shot = new Shot();
				int dx, dy;
				//Point _tpt = pigs.get(0).getCenter();
				
				releasePoint = tp.findReleasePoint(sling, releaseAngle);
				Point refPoint = tp.getReferencePoint(sling);
				
				if (releasePoint != null){
					double cal_releaseAngle = tp.getReleaseAngle(sling, releasePoint);
					System.out.println("Release Point: " + releasePoint);
					System.out.println("Release Angle: "
							+ Math.toDegrees(cal_releaseAngle));
								
					
					List<Point> traj = tp.predictTrajectory(sling, releasePoint);
					Point tpt = null;
					for (Point p : traj){
						int px = (int)p.getX();
						int py = (int)p.getY();
						if (px>250 && px<sc_w && py<sc_h && py>0){
							int pix = img.getRGB((int)p.getX(), (int)p.getY());
							int r = (pix >> 16) & 0xFF;
							int g = (pix >> 8) & 0xFF;
							int b = pix & 0xFF;
							if (r != 0 && g != 0 && b != 0){
								System.out.println("p: "+p.getX()+","+p.getY() +" "+ r+" "+g+" "+b);
								tpt = p;
								break;
							}
							//img.setRGB((int)p.getX(), (int)p.getY(), 0xff0000);
						}
					}
					//File f = new File ("asdf.png");
					//ImageIO.write(img, "PNG", f);
					
					int tapInterval = 0;
					int tapTime = 0;
					
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
					
					if (tpt != null){
						tapTime = tp.getTapTime(sling, releasePoint, tpt, tapInterval);
						System.out.println(tpt.getX() + ","+tpt.getY());
						System.out.println("tapTime: "+tapTime);
					}
					//int tapInterval = 0;
					//int tapTime = tp.getTapTime(sling, releasePoint, _tpt, tapInterval);
					dx = (int)releasePoint.getX() - refPoint.x;
					dy = (int)releasePoint.getY() - refPoint.y;
					shot = new Shot(refPoint.x, refPoint.y, dx, dy, 0, tapTime);
				}
				else{
					System.err.println("No Release Point Found");
					return state;
				}
				
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
	
	
	public GameState solve() throws IOException
	{

		// capture Image
		BufferedImage screenshot = ActionRobot.doScreenShot();
		
		// 2017-04-01 : ymkim1019
		System.out.println("screen shot size = " + screenshot.getWidth() + "," + screenshot.getHeight());
		System.out.println(aRobot.getState());
		// process image
		Vision vision = new Vision(screenshot);

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
				send_env_to_agent(vision);
				
				int size = in.readInt();
				int job_id = in.readInt();
				int temp = in.readInt();
				System.out.format("size=%d, job_id=%d, data=%d\n", size, job_id, temp);
				
				Point releasePoint = null;
				Shot shot = new Shot();
				int dx,dy;
				
				{
					// random pick up a pig
					ABObject pig = pigs.get(randomGenerator.nextInt(pigs.size()));
					
					Point _tpt = pig.getCenter();// if the target is very close to before, randomly choose a
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

						int tapTime = tp.getTapTime(sling, releasePoint, _tpt, tapInterval);
						System.out.println("tapTime: "+tapTime);
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

				// ymkim1019 below lines are commented to speed up the game progress
				
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
