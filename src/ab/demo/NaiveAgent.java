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
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

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
	private int shot_num;
	private int[] episode_num;
	private Point prevTarget;
	private int global_episode;
	private ABType[] birdTypeArray;
	private List<List<List<Integer>>> history;
	private int nohope;
	private int prevpignum;
	private int currentpignum;
	
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
		//firstShot = true;
		episode_num = new int[10];
		global_episode = 0;
		shot_num = 1;
		randomGenerator = new Random();
		history = new ArrayList<List<List<Integer>>>();
		nohope = 0;
		prevpignum = -1;
		currentpignum = 0;
		// --- go to the Poached Eggs episode level selection page ---
		ActionRobot.GoFromMainMenuToLevelSelection();

	}
	
	// run the client
	public void run() {
		// 2017-04-01 : ymkim1019
		// try {
		// 	so = new Socket(agent_ip, agent_port);
		// 	System.out.println("Connected to the Agent..");
		// 	in = new DataInputStream(so.getInputStream());
		// 	out = new DataOutputStream(so.getOutputStream());
		// } catch (Exception e) {
		// 	System.out.println("Fail to connect to the Agent..");
		// 	return;
		// }
		
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
				System.out.println("win!!!!!!!");
				try {
					Thread.sleep(3000);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				// 2017-04-01 : ymkim1019
				// Update the current stage score and stars
				int score = StateUtil.getScore(ActionRobot.proxy);
				save_score(score);
				//int stars = StateUtil.getStars(ActionRobot.proxy);

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
							+ " Score: " + scores.get(key));
				}
				System.out.println("Total Score: " + totalScore);
				if(currentLevel<9){
					currentLevel++;
				}
				else{
					currentLevel = 1;
				}
				aRobot.loadLevel(currentLevel);
				// make a new trajectory planner whenever a new level is entered
				tp = new TrajectoryPlanner();

				// first shot on this level, try high shot first
				//firstShot = true;
				shot_num = 1;
			} else if (state == GameState.LOST) {

				save_score(0);
				System.out.println("Restart");
				aRobot.restartLevel();
				shot_num = 1;
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

	public void save_score(int score)
	{
		//Writer jsonfw = new FileWriter(".\\history.json");

		List<List<Integer>> episode;
		episode = history.get(history.size() - 1);

		List<Integer> transition = new ArrayList<Integer>(Arrays.asList(score));
		episode.add(transition);
		//jsonfw.close();
	}

	public void save_history(Vision vision,int sling_x,int sling_y, int width, int height,int num_pig, int num_block, int num_bird,int birdtype, int x,int y,int dx,int dy,int taptime) throws IOException
	{
		File outputfile = new File(String.format("D:\\angrybird_image\\%d_%d_%d.jpg",currentLevel,episode_num[currentLevel-1],shot_num));
		Writer jsonfw = new FileWriter(".\\history.json");
		BufferedImage imgBuf = vision.getImageBuffer();
		Preprocessor prep = new Preprocessor(imgBuf);
		imgBuf = prep.drawImage(imgBuf, false);
		ImageIO.write(imgBuf, "jpg", outputfile);
		//List<List<List<Integer>>> history = new ArrayList<List<List<Integer>>>();
		List<List<Integer>> episode;
		if(shot_num == 1){
			episode = new ArrayList<List<Integer>>();
			history.add(episode);
		}
		else{
			episode = history.get(history.size() - 1);
		}

		List<Integer> transition = new ArrayList<Integer>(Arrays.asList(currentLevel, episode_num[currentLevel-1], shot_num, sling_x, sling_y, width, height, num_pig, num_block, num_bird, birdtype, x, y, dx, dy, taptime));
		episode.add(transition);


		Gson gson = new GsonBuilder().create();
    String jsonstr = gson.toJson(history);
    jsonfw.write(jsonstr);
    jsonfw.close();
    //jsonfw.close();

	}

	public GameState solve() throws IOException
	{

		// capture Image
		BufferedImage screenshot = ActionRobot.doScreenShot();
		
		// 2017-04-01 : ymkim1019
		//System.out.println("screen shot size = " + screenshot.getWidth() + "," + screenshot.getHeight());

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
    	List<ABObject> blocks = vision.findBlocksMBR();
    	List<ABObject> birds = vision.findBirdsMBR();

   		//  if(shot_num == 1){
 		// 	BufferedImage imgBuf = vision.getImageBuffer();
			// Preprocessor prep = new Preprocessor(imgBuf);
			// imgBuf = prep.drawImage(imgBuf, false);

			// birdTypeArray = prep.getBirds();
 		// }
    	int birdtype;

		GameState state = aRobot.getState();

		// if there is a sling, then play, otherwise just skip.
		if (sling != null) {

			if (!pigs.isEmpty()) {
				// 2017-05-07 : ymkim1019
				// send_env_to_agent(vision,sling.x,sling.y,sling.width,sling.height,pigs.size(),blocks.size(),birds.size(),birdtpye);
				
				// int size = Integer.reverseBytes(in.readInt());
				// int job_id = Integer.reverseBytes(in.readInt());
				// int dx = Integer.reverseBytes(in.readInt());
				// int dy = Integer.reverseBytes(in.readInt());
				// int tapint = Integer.reverseBytes(in.readInt());
				// System.out.format("size=%d, job_id=%d, data=%d\n", size, job_id, dx);

				Point releasePoint = null;
				Shot shot = new Shot();
				int dx,dy;
				int tapTime;
				{
					// random pick up a pig
					ABObject pig = pigs.get(randomGenerator.nextInt(pigs.size()));
					
					Point _tpt = pig.getCenter();// if the target is very close to before, randomly choose a
					// point near it
					// if (prevTarget != null && distance(prevTarget, _tpt) < 10) {
					// 	double _angle = randomGenerator.nextDouble() * Math.PI * 2;
					// 	_tpt.x = _tpt.x + (int) (Math.cos(_angle) * 10);
					// 	_tpt.y = _tpt.y + (int) (Math.sin(_angle) * 10);
					// 	System.out.println("Randomly changing to " + _tpt);
					// }
					int delta = randomGenerator.nextInt(101) - 50;
					// if(shot_num <2){
					// 	delta = 100;
					// }
					// if(shot_num ==2){
					// 	delta = -30;
					// }
					_tpt.x += delta;
					_tpt.y -= delta;
					prevTarget = new Point(_tpt.x, _tpt.y);

					// estimate the trajectory
					ArrayList<Point> pts = tp.estimateLaunchPoint(sling, _tpt);
					
					// 2017-04-01 : ymkim1019
					System.out.println("# of launch points=" + pts.size());
					
					// do a high shot when entering a level to find an accurate velocity
					// if ((shot_num == 1) && pts.size() > 1) 
					// {
					// 	releasePoint = pts.get(1);
					// }
					// else 
					if (pts.size() == 1)
						releasePoint = pts.get(0);
					else if ((pts.size() == 2)) //&& (birdTypeArray.length != shot_num))
					{
						// randomly choose between the trajectories, with a 1 in
						// 6 chance of choosing the high one
						if (randomGenerator.nextInt(5) == 0)
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
						ABType birdtypeT = aRobot.getBirdTypeOnSling();
						birdtype = birdtypeT.ordinal();
						
						//birdtype = birdTypeArray[shot_num-1].ordinal();
						switch (birdtypeT) 
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

						tapTime = tp.getTapTime(sling, releasePoint, _tpt, tapInterval);
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
								//send_act_to_agent(currentLevel,shot_num,prevTarget.x,prevTarget.y,shot.getDx(),shot.getDy(),shot.getT_tap());
								if (shot_num == 1){
									episode_num[currentLevel-1]++;
									global_episode++;
								}
								prevpignum = currentpignum;
							    currentpignum = pigs.size();
							    if(currentpignum == prevpignum){
							    	nohope++;
							    } 
							    else{
							    	nohope =0;
							    }
							    if(nohope == 2){
							    	currentpignum = -1;
							    	nohope=0;
							    	return GameState.LOST;
							    }
								save_history(vision,sling.x,sling.y, sling.width, sling.height,pigs.size(), blocks.size(), birds.size(), birdtype, prevTarget.x, prevTarget.y, dx, dy, tapTime);
								aRobot.cshoot(shot);
								state = aRobot.getState();
								shot_num++;
								// if ( state == GameState.PLAYING )
								// {
								// 	screenshot = ActionRobot.doScreenShot();
								// 	vision = new Vision(screenshot);
								// 	List<Point> traj = vision.findTrajPoints();
								// 	// 2017-04-01 : ymkim1019
								// 	// below codes calibrate the trajectory module
								// 	tp.adjustTrajectory(traj, sling, releasePoint);
								// 	//firstShot = false;
								// }
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
