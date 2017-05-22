// 2017-05-19 : jyham

package ab.vision;

import java.awt.Color;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
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
	
	public List<ABObject> getBirds(){
		List<ABObject> birds = new LinkedList<ABObject>();
		return birds;
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
}
