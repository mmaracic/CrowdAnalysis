/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hr.mmaracic.crowdanalysis;

import hr.mmaracic.opencvdemo.Util;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.util.Collections;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2GRAY;


/**
 *
 * @author FEANOR-ROG
 */
public class CrowdAnalysis {
    
    public static void main(String[] args){
        // load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
        //substring 1 to remove leading "/"
        String fileName = ClassLoader.getSystemClassLoader().getResource("crowd.jpg").getFile().substring(1);
        Mat img = Imgcodecs.imread(fileName);
        if (img.width() == 0 && img.height() ==0){
            throw new IllegalArgumentException("Incorrect path, dimensions of the image are 0");
        }
        Image i = Util.toBufferedImage(img);
        Graphics g = i.getGraphics();
        Util.displayImage(i, "Crowd image");   
        
        Mat gray = new Mat(i.getHeight(null), i.getWidth(null), CvType.CV_8U);
        Imgproc.cvtColor(img, gray, COLOR_RGB2GRAY);

        Image i2 = Util.toBufferedImage(gray);
        Graphics g2 = i2.getGraphics();
        Util.displayImage(i2, "Grayscale image");
        
        Mat corners = new Mat(gray.rows(), gray.cols(), CvType.CV_32FC1);
        Imgproc.cornerHarris(gray, corners, 2, 3, 0.04);
        Core.MinMaxLocResult minMax = Core.minMaxLoc(corners);

        Mat result = img.clone();
        Image i3 = Util.toBufferedImage(result);
        Graphics g3 = i3.getGraphics();
        Mat scaledCorners = new Mat(corners.rows(), corners.cols(), CvType.CV_8U);
        double multiplier = 255.0/(minMax.maxVal - minMax.minVal);
        for(int y=0;y<corners.rows();y++){
            for(int x=0;x<corners.cols();x++){
                Double dValue = (corners.get(y, x)[0]-minMax.minVal)*multiplier;
                scaledCorners.put(y, x, new byte[]{dValue.byteValue()});
                if (dValue > 160){
                    g3.setColor(Color.red);
                    g3.drawOval(x, y, 3, 3);                    
                }
            }
        }
        Util.displayImage(i3, "Harris corners");
        
        Mat hist = new Mat();
        Imgproc.calcHist(Collections.singletonList(scaledCorners), new MatOfInt(0), new Mat(), hist, new MatOfInt(256), new MatOfFloat(0,256));
        System.out.println(hist.dump());
    }
}
