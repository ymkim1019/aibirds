// 2017-05-19 : jyham

package ab.vision;

import java.awt.Color;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

public class Preprocessor {	
	private static List<Rectangle> pigs, redBirds, blueBirds, yellowBirds, blackBirds, whiteBirds, iceBlocks, woodBlocks, stoneBlocks, TNTs;
	private static VisionRealShape visionRealShape;

    // detected game objects
    private Rectangle _sling = null;
    private List<ABObject> _birds = null;
    
	public Preprocessor(BufferedImage rawimg){
		visionRealShape = new VisionRealShape(rawimg);
		visionRealShape.findObjects();
		visionRealShape.findPigs();
		visionRealShape.findHills();
		_birds = visionRealShape.findBirds();
		_sling = visionRealShape.findSling();
	}
	
	// 2017-06-03 : jyham
	public ABType[] getBirds(){
		if(_sling == null) return null;
		if(_birds.isEmpty()) return null;
		
		double sling_position = _sling.getCenterX();
		int birds_num = _birds.size();
		ABType[] birds_seq = new ABType[birds_num];
		
		if (birds_num == 1){
			birds_seq[0] = _birds.get(0).getType();
			return birds_seq;
		}
		
		Collections.sort(_birds, new Comparator<Rectangle>(){
			@Override
			public int compare(Rectangle o1, Rectangle o2) {
				return o1.getCenterX() > o2.getCenterX() ? -1
						: o1.getCenterX() < o2.getCenterX() ? 1 : 0;
			}	
		});
		
		int i = 0;
		if (_birds.get(1).getCenterX() > sling_position){
			birds_seq[0] = _birds.get(birds_num-1).getType();
			_birds.remove(birds_num-1);
			i = 1;
		}
		for (ABObject b : _birds){
			birds_seq[i] = b.getType();
			i++;
		}
		return birds_seq;
	}
	
	public BufferedImage drawImage(BufferedImage rawimg, boolean draw_orig_obj){
		// get game state
		GameStateExtractor game = new GameStateExtractor();
		GameStateExtractor.GameState state = game.getGameState(rawimg);
		if (state != GameStateExtractor.GameState.PLAYING) 
		{
			rawimg = VisionUtils.convert2grey(rawimg);
			return rawimg;
		}
		return visionRealShape.drawFillObjects();
	}
	public Rectangle getSling(){
		return _sling;
	}
	
	public int getGround(){
		return visionRealShape.getGround();
	}
	
}
