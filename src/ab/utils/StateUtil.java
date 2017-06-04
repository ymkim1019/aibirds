/*****************************************************************************
** ANGRYBIRDS AI AGENT FRAMEWORK
** Copyright (c) 2014,XiaoYu (Gary) Ge, Stephen Gould,Jochen Renz
**  Sahan Abeyasinghe, Jim Keys,   Andrew Wang, Peng Zhang
** All rights reserved.
**This work is licensed under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
**To view a copy of this license, visit http://www.gnu.org/licenses/
*****************************************************************************/

package ab.utils;

import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;

import ab.server.Proxy;
import ab.server.proxy.message.ProxyScreenshotMessage;
import ab.vision.GameStateExtractor;
import ab.vision.VisionUtils;
import ab.vision.GameStateExtractor.GameState;
//import ab.vision.GameStateExtractor.RectLeftOf;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;


public class StateUtil {
    /**
     * Get the current game state
     * @return GameState: the current state 
     * */
	public static  GameState  getGameState(Proxy proxy)
	{
		 byte[] imageBytes = proxy.send(new ProxyScreenshotMessage());
		
	        BufferedImage image = null;
	        try {
	            image = ImageIO.read(new ByteArrayInputStream(imageBytes));
	        } catch (IOException e) {
	          
	        }
	        GameStateExtractor gameStateExtractor = new GameStateExtractor();
	        GameStateExtractor.GameState state = gameStateExtractor.getGameState(image);
		  return state;
	}

	private static int _getStars(Proxy proxy)
	{
		byte[] imageBytes = proxy.send(new ProxyScreenshotMessage());
        int stars = 0;

		BufferedImage image = null;
	    try {
	           image = ImageIO.read(new ByteArrayInputStream(imageBytes));
	        } 
	    catch (IOException e) {
	           e.printStackTrace();
	        }
	    
        GameStateExtractor gameStateExtractor = new GameStateExtractor();
        GameState state = gameStateExtractor.getGameState(image);
        if (state == GameState.PLAYING)
        	stars = 0;
        else
        	if(state == GameState.WON)
        	{
        		// crop score image
        		BufferedImage starImage = image.getSubimage(270, 180, 300, 32);
        		int color1 = starImage.getRGB(50, 19);
        		int color2 = starImage.getRGB(156, 17);
        		int color3 = starImage.getRGB(245, 25);
        		int mask1 = (((color1 & 0x00ff0000) >> 16) > 192) ? 1 : 0;
        		int mask2 = (((color2 & 0x00ff0000) >> 16) > 192) ? 1 : 0;
        		int mask3 = (((color3 & 0x00ff0000) >> 16) > 192) ? 1 : 0;
        				
        		stars = mask1 + mask2 + mask3;
        		
        		System.out.printf("stars : %d, %d, %d\n", mask1, mask2, mask3);

        		/*
        		File outputfile = new File("stars.png");
        		try {
					ImageIO.write(starImage, "PNG", outputfile);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				*/
        	}
        		
        if(stars == 0)
    	   System.out.println(" Starts is unavailable "); 	   
        return stars;
	}
	
	private static int _getScore(Proxy proxy)
	{
		byte[] imageBytes = proxy.send(new ProxyScreenshotMessage());
        int score = -1;

		BufferedImage image = null;
	    try {
	           image = ImageIO.read(new ByteArrayInputStream(imageBytes));
	        } 
	    catch (IOException e) {
	           e.printStackTrace();
	        }
	    
        GameStateExtractor gameStateExtractor = new GameStateExtractor();
        GameState state = gameStateExtractor.getGameState(image);
        if (state == GameState.PLAYING)
        	score = gameStateExtractor.getScoreInGame(image);
        else
        	if(state == GameState.WON)
        		score = gameStateExtractor.getScoreEndGame(image);
       if(score == -1)
    	   System.out.println(" Game score is unavailable "); 	   
		return score;
	}
	/**
	 * The method checks the score every second, and return when the score is stable (not flashing).
	 * 
	 * @return score: the current score.
	 * 
	 * */
	public static int getScore(Proxy proxy)
	{

		int current_score = -1;
		while (current_score != _getScore(proxy)) 
		{
		  try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
			
				e.printStackTrace();
			}
		   if(getGameState(proxy) == GameState.WON)
		   {	
			   current_score = _getScore(proxy);
		   }
		   else
			   System.out.println(" Unexpected state: PLAYING");
		}
		return current_score;
	}
	
	public static int getStars(Proxy proxy)
	{

		int current_score = -1;
		while (current_score != _getScore(proxy)) 
		{
		  try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
			
				e.printStackTrace();
			}
		   if(getGameState(proxy) == GameState.WON)
		   {	
			   current_score = _getScore(proxy);
		   }
		   else
			   System.out.println(" Unexpected state: PLAYING");
		}
		
		int current_stars = _getStars(proxy);
		
		return current_stars;
	}
}
