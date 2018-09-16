 package cz.nakoncisveta.eyetracksample.eyedetect;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.ProgressBar;
import android.widget.ToggleButton;
import android.widget.CompoundButton;
import android.widget.CompoundButton.OnCheckedChangeListener;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;


import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.Object;

import java.util.Date;
import android.widget.TimePicker;
import java.util.Calendar;
import java.util.Timer;
import java.util.TimerTask;
import android.os.Handler;
import android.os.Message;
import android.os.Handler.Callback;

import android.os.CountDownTimer;
import android.widget.CompoundButton;
import android.widget.CompoundButton.OnCheckedChangeListener;
import android.widget.Toast;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.content.Context;
import android.content.Intent;


 public class FdActivity extends Activity implements CvCameraViewListener2 {


//    private MyCountDownTimer myCountDownTimer;
//CountDownTimer countdowntimer;
//    private final long startTime = 10 * 1000;
//    private final long interval = 1 * 1000;
    private Handler handler;
    private Runnable runnable;
    private CountDownTimer cdt;
    private int time = 0;
    private Boolean timerbb = false;
    private Button btnCount;
    private boolean timerStarted = false;
    private boolean timerStarted1 = false;
    private TimePicker timePicker1;


    private static final String    TAG                 = "OCVSample::Activity";
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    public static final int        JAVA_DETECTOR       = 0;
    private static final int TM_SQDIFF = 0;
    private static final int TM_SQDIFF_NORMED = 1;
    private static final int TM_CCOEFF = 2;
    private static final int TM_CCOEFF_NORMED = 3;
    private static final int TM_CCORR = 4;
    private static final int TM_CCORR_NORMED = 5;

    private Boolean c = false;
    private Boolean o = false;
    private Boolean cc = false;
    private Boolean oo = false;
    private Boolean ccc = false;
    private Boolean ooo = false;
    private int countblink;
    private Boolean face = true;

    private Mat templateR_open;
    Point iris;
    Rect eye_template;
    Core.MinMaxLocResult mmG;
    private boolean HaarEyeOpen_R = false;
    private Rect maxRect;

    private int learn_frames = 0;
    private Mat teplateR;
    private Mat teplateL;
    int method = 0;
    private Mat RR;
    private Mat LL;
    private Mat EyeR;
    private Mat EyeL;

    // matrix for zooming
    private Mat mZoomWindow;
    private Mat mZoomWindow2;

    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;

    private Mat                    mRgba;
    private Mat                    mGray;
    private Mat                    mGrays;
    private File                   mCascadeFile;
    private File                   mCascadeFileEye;
    private CascadeClassifier      mJavaDetector;
    private CascadeClassifier      mJavaDetectorEye;


    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.2f;
    private int mAbsoluteFaceSize = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;
    //private SeekBar mMethodSeekbar;
    //private TextView mValue;

    double xCenter = -1;
    double yCenter = -1;

     MediaPlayer beep;

     double frequency;
     long timer;
     long timer1;
     long timer2;
     long timer3;
     boolean flag = false;


    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");


                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        // load cascade file from application resources
                        InputStream ise = getResources().openRawResource(R.raw.haarcascade_righteye_2splits);
                        File cascadeDirEye = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFileEye = new File(cascadeDirEye, "haarcascade_righteye_2splits.xml");
                        FileOutputStream ose = new FileOutputStream(mCascadeFileEye);

                        while ((bytesRead = ise.read(buffer)) != -1) {
                            ose.write(buffer, 0, bytesRead);
                        }
                        ise.close();
                        ose.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mJavaDetectorEye = new CascadeClassifier(mCascadeFileEye.getAbsolutePath());
                        if (mJavaDetectorEye.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier for eye");
                            mJavaDetectorEye = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFileEye.getAbsolutePath());

                        cascadeDir.delete();
                        cascadeDirEye.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
                    mOpenCvCameraView.enableFpsMeter();
                    mOpenCvCameraView.setCameraIndex(1);
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }


    };

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {

        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);


        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);

        //contTextView = new TextView(this);


//        this.setContentView(contTextView);


//        mMethodSeekbar = (SeekBar) findViewById(R.id.methodSeekBar);
//        mValue = (TextView) findViewById(R.id.method);

        beep = MediaPlayer.create(this, R.raw.button1);

//        buttonStart = (Button)findViewById(R.id.s_b);

        //mMethodSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {

//            @Override
//            public void onStopTrackingTouch(SeekBar seekBar)
//            {
//                // TODO Auto-generated method stub
//
//            }
//
//            @Override
//            public void onStartTrackingTouch(SeekBar seekBar)
//            {
//                // TODO Auto-generated method stub
//
//            }

//            @Override
//            public void onProgressChanged(SeekBar seekBar, int progress,
//                                          boolean fromUser)
//            {
//                method = progress;
//                switch (method) {
//                    case 0:
//                        mValue.setText("TM_SQDIFF");
//                        break;
//                    case 1:
//                        mValue.setText("TM_SQDIFF_NORMED");
//                        break;
//                    case 2:
//                        mValue.setText("TM_CCOEFF");
//                        break;
//                    case 3:
//                        mValue.setText("TM_CCOEFF_NORMED");
//                        break;
//                    case 4:
//                        mValue.setText("TM_CCORR");
//                        break;
//                    case 5:
//                        mValue.setText("TM_CCORR_NORMED");
//                        break;
//                }
//
//
//            }
//        });
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mGrays = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mGrays.release();
        mRgba.release();
//        mZoomWindow.release();
//        mZoomWindow2.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        frequency = Core.getTickFrequency(); //frecuency of the clock. How many clocks cycles per second,

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        mGrays = inputFrame.gray();
        Size sf = new Size(480,270);
        Imgproc.resize(mGray, mGray, sf);
//        Imgproc.resize(mRgba, mRgba, sf);


/*//        Imgproc.threshold(mGray, mRgba, 20, 255, Imgproc.THRESH_BINARY_INV);
        Imgproc.GaussianBlur(mGray, mGray, new Size(15, 15), 50);
        Imgproc.equalizeHist(mGray, mGray);
        Imgproc.Sobel(mGray, mRgba,CvType.CV_16S, 0, 1);
//        Imgproc.Sobel(mGray, mRgba,CvType.CV_16S, 1, 0);

        Core.convertScaleAbs(mRgba, mRgba);
        Core.normalize(mRgba,mRgba,0,255,Core.NORM_MINMAX);
        Core.addWeighted(mRgba, 0.5, mRgba, 0.5, -0.5, mRgba );
        Imgproc.threshold(mRgba, mRgba, 25, 255, Imgproc.THRESH_BINARY);
//        Imgproc.medianBlur(mRgba, mRgba, 11);*/


        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
               }

            }

            if (mZoomWindow == null || mZoomWindow2 == null)
                CreateAuxiliaryMats();

            MatOfRect faces = new MatOfRect();

            if (mDetectorType == JAVA_DETECTOR) {
                if (mJavaDetector != null)
                    mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                            new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
            } else {
                Log.e(TAG, "Detection method is not selected!");
            }

            Rect[] facesArray = faces.toArray();


            for (int i = 0; i < facesArray.length; i++) {

                int x = (facesArray[0].x)*4;
                int y = (facesArray[0].y)*4;
                int width = (facesArray[0].width)*4;
                int height = (facesArray[0].height)*4;


//                Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
                Imgproc.rectangle(mRgba, new Point (x,y), new Point (x+width,y+height), FACE_RECT_COLOR, 3);
                xCenter = (x + width + x) / 2;
                yCenter = (y + y + height) / 2;
                Point center = new Point(xCenter, yCenter);



//                Imgproc.circle(mRgba, center, 10, new Scalar(255, 0, 0, 255), 3); //วาดวงกลม

//                Imgproc.putText(mRgba, "[" + center.x + "," + center.y + "]",new Point(center.x + 20, center.y + 20),Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));

//                Rect r = new Rect (facesArray[0].x, facesArray[0].y,facesArray[0].width, facesArray[0].height);
                Rect r = new Rect ((facesArray[0].x)*4, (facesArray[0].y)*4,(facesArray[0].width)*4, (facesArray[0].height)*4);
//                Rect r = facesArray[0];


                // compute the eye area
                Rect eyearea = new Rect(r.x + r.width / 32,
                        (int) (r.y + (r.height / 4.5)),
                        r.width - 2 * r.width / 32,
                        (int) (r.height / 3.0));


                // split it
                Mat eye = new Mat();
                Imgproc.GaussianBlur(mGrays, mGrays, new Size(15, 15), 50);
                Imgproc.equalizeHist(mGrays, mGrays);
                Imgproc.Sobel(mGrays, eye,CvType.CV_16S, 0, 1);
                Core.convertScaleAbs(eye, eye);
                Core.normalize(eye,eye,0,255,Core.NORM_MINMAX);
                Core.addWeighted(eye, 0.5, eye, 0.5, -0.5, eye );
                Imgproc.threshold(eye, eye, 25, 255, Imgproc.THRESH_BINARY);


                Rect eyearea_rightt = new Rect(r.x + r.width / 4,
                        (int) (r.y + (r.height / 2.9)),
                        (r.width - 2 * r.width / 6) / 10,
                        (int) (r.height / 50));
                Rect eyearea_right = new Rect((r.x + r.width / 6) + 25,
                    (int) (r.y + (r.height / 3.2)),
                    (r.width - 2 * r.width / 6) / 3,
                    (int) (r.height / 7));

                Mat RR = new Mat(eye, eyearea_rightt);

                Size szR = new Size(10, 10);
                Imgproc.resize(RR, RR, szR);


                int nr = Core.countNonZero(RR);

//            if (nr>1000)
//            { Imgproc.putText(mRgba,"Open",new Point(center.x - 150, center.y + 20),
//                    Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255, 255, 255, 255));}
//            else
//            { Imgproc.putText(mRgba,"close",new Point(center.x - 150, center.y + 20),
//                    Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255, 255, 255, 255));}
//            Imgproc.putText(mRgba,Integer.toString(nr),new Point(center.x + 20, center.y + 20),
//                    Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255, 255, 255, 255));


//            for (int rx = 0; rx < sizeR.height; rx++) {
//                for (int ry = 0; ry < sizeR.width; ry++) {
//                    nr = RR.get(rx, ry);
//                    pr[rx] = pr[rx] + nr;
//                }
//
//            }

//            double[] nr = RR.get(10, 10);

//                Imgproc.putText(mRgba, Integer.toString(nr), new Point(center.x  -250, center.y - 20),
//                        Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255, 255, 255, 255));

                Rect eyearea_leftt = new Rect(r.x + r.width / 64 + (r.width + 10 * r.width / 32) / 2,
                        (int) (r.y + (r.height / 2.9)),
                        (r.width - 2 * r.width / 6) / 10,
                        (int) (r.height / 50));
                Rect eyearea_left = new Rect  (r.x + r.width / 64 + (r.width + 4 * r.width / 32) / 2,
                    (int) (r.y + (r.height / 3.2)),
                    (r.width - 2 * r.width / 4) / 2,
                    (int) (r.height / 7));


                Mat LL = new Mat(eye, eyearea_leftt);
                Size szL = new Size(10, 10);
                Imgproc.resize(LL, LL, szL);

                int nl = Core.countNonZero(LL);

                //int rowsL = LL.rows();
                //int colsL = LL.cols();
                //int count_white;
                //int ch = LL.channels(); //Calculates number of channels (Grayscale: 1, RGB: 3, etc.)
//            if (nl>1000)
//            { Imgproc.putText(mRgba,"Open",new Point(center.x - 150, center.y + 20),
//                    Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(0, 0, 0, 0));}
//            else
//            { Imgproc.putText(mRgba,"close",new Point(center.x - 150, center.y + 20),
//                    Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(0, 0, 0, 0));}

//            Imgproc.putText(mRgba,Integer.toString(nl),new Point(center.x + 150, center.y -20),
//                    Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255, 255, 255, 255));

//                Imgproc.putText(mRgba, Integer.toString(nl), new Point(center.x - 100, center.y + 20), Core.FONT_HERSHEY_SIMPLEX, 1, new Scalar(255, 255, 255, 255));

//                Rect rectR = get_template(mJavaDetectorEye, eyearea_right);
//                if (rectR.width==0 || rectR.height==0){continue;}
//
//                rectR = get_template(mJavaDetectorEye, rectR, new Size(1, 1), new Size(50,50));
//                templateR_open = mGray.submat(rectR);
//                HaarEyeOpen_R = match_eye(templateR_open);

//                if(!HaarEyeOpen_R){
//                    cc = true;
//                }
//                else{
//                    oo = true;
//                }

//                if(o == true && oo == true)
//                {
//                    ooo = true;
//                }
//
//                if(c == true && cc == true)
//                {
//                    ccc = true;
//                }

/*                Rect rectR = get_template(mJavaDetectorEye, eyearea_right);
                rectR = get_template(mJavaDetectorEye, rectR, new Size(1, 1), new Size(50,50));
                templateR_open = mGray.submat(rectR);*/
//&& templateR_open != null

                if (face == true) {
                    if (nl != 0 && nr != 0) {
                        o = true;
                        timerStarted1 = false;
//                        Imgproc.putText(mRgba,"Open", new Point(center.x - 50, center.y + 20),
//                            Core.FONT_HERSHEY_SIMPLEX, 4, new Scalar(0, 0, 0, 0));
                    } else if (nl == 0 && nr == 0){
                        c = true;
//                        Imgproc.putText(mRgba,"Closed", new Point(center.x - 50, center.y + 20),
//                            Core.FONT_HERSHEY_SIMPLEX, 4, new Scalar(0, 0, 0, 0));
                        if(!timerStarted1)
                        {
                            timer2 = Core.getTickCount();
                            timerStarted1 = true;
                        }
                        timer3 = Core.getTickCount();

                        if((timer3-timer2)/frequency > 4)
                        {
                            flag = true;
                            while(flag)
                            {
                                beep.start();
                                Imgproc.putText(mRgba,"ALERT", new Point(center.x - 20, center.y + 20),
                                    Core.FONT_HERSHEY_SIMPLEX, 10, new Scalar(255, 0, 0, 255));
                                return  mRgba;

                            }
                            timerStarted1 = false;
                        }

                    }
                    if (o == true && c == true) {
                        countblink = countblink + 1;
                        o = false;
                        c = false;
                        cc = true;
//                      oo = false;
//                      cc = false;

                    }
                    Imgproc.putText(mRgba, Integer.toString(countblink/2), new Point(center.x - 50, center.y + 70),
                            Core.FONT_HERSHEY_SIMPLEX, 4, new Scalar(0, 0, 0, 0));
                }
                else{
                    o = false;
                    c = false;
                    cc = true;
                    countblink = countblink - 1;
                }

//                if(cc){
//                    countblink = countblink - 1;
//                    cc = false;
//                }


                if((countblink/2)>16)
                {
                    flag = true;
                    while(flag)
                    {
                        beep.start();
                        Imgproc.putText(mRgba,"ALERT", new Point(center.x - 20, center.y + 20),
                                Core.FONT_HERSHEY_SIMPLEX, 10, new Scalar(255, 0, 0, 255));
                        return  mRgba;
                    }
                }


//            Imgproc.putText(mRgba,Integer.toString(countblink/4),new Point(center.x - 50, center.y + 50),
//                    Core.FONT_HERSHEY_SIMPLEX, 4, new Scalar(255, 255, 255, 255));

//            new CountDownTimer(10000, 1000) { //10 s
//
//                public void onTick(long millisUntilFinished) {
//                    int time = (int) millisUntilFinished;
//                    Imgproc.putText(mRgba,Integer.toString(time /1000),new Point(0, 0 ),Core.FONT_HERSHEY_SIMPLEX, 1, new Scalar(255, 255, 255, 255));
//                    //mTextField.setText("seconds remaining: " + millisUntilFinished / 1000);
//                }
//
//                public void onFinish() {
//                    Imgproc.putText(mRgba,"ok",new Point(20,20),Core.FONT_HERSHEY_SIMPLEX, 1, new Scalar(255, 255, 255, 255));
//                }
//            }.start();


                // draw the area - mGray is working grayscale mat, if you want to
                // see area in rgb preview, change mGray to mRgba
                Imgproc.rectangle(mRgba, eyearea_left.tl(), eyearea_left.br(),
                        new Scalar(255, 0, 0, 255), 2);
                Imgproc.rectangle(mRgba, eyearea_right.tl(), eyearea_right.br(),
                        new Scalar(255, 0, 0, 255), 2);

                Imgproc.rectangle(mRgba, eyearea_leftt.tl(), eyearea_leftt.br(),
                        new Scalar(255, 255, 0, 0), 2);
                Imgproc.rectangle(mRgba, eyearea_rightt.tl(), eyearea_rightt.br(),
                        new Scalar(255, 255, 0, 0), 2);

            /*if (learn_frames < 5) {
                teplateR = get_template(mJavaDetectorEye, eyearea_right, 12);
                teplateL = get_template(mJavaDetectorEye, eyearea_left, 12);
                learn_frames++;
            } else {
                // Learning finished, use the new templates for template
                // matching
                match_eye(eyearea_right, teplateR, method);
                match_eye(eyearea_left, teplateL, method);

            }*/


                // cut eye areas and put them to zoom windows
                //Imgproc.resize(mRgba.submat(eyearea_left), mZoomWindow2, mZoomWindow2.size());
                //Imgproc.resize(mRgba.submat(eyearea_right), mZoomWindow, mZoomWindow.size());
                //Imgproc.resize(mRgba.submat(eyearea_left), mZoomWindow2, mZoomWindow2.size());
                //Imgproc.resize(mRgba.submat(eyearea_right), mZoomWindow, mZoomWindow.size());

//                MyTimerTask myTask = new MyTimerTask();
//                Timer myTimer = new Timer();
//                myTimer.schedule(myTask, 3000);

//                int min = timePicker1.getCurrentMinute();
                //if (min==min+1)

//                frequency = Core.getTickFrequency(); //frecuency of the clock. How many clocks cycles per second,
//                timer = Core.getTickCount();			//start timer for 1 minute. It gives number of clock cycles.
//                frequency1 = Core.getTickFrequency(); //frecuency of the clock. How many clocks cycles per second,
//                timer1 = Core.getTickCount();			//start timer for 1 minute. It gives number of clock cycles.
//                frequency2 = Core.getTickFrequency(); //frecuency of the clock. How many clocks cycles per second,
//                timer2 = Core.getTickCount();			//start timer for 1 minute. It gives number of clock cycles.

//                if (timerbb == true) {
//                    int countblink = 0;
//                    timerbb = false;
//                }




//                Timer timer = new Timer();
//                timer.schedule(new TimerTask() {
//                    @Override
//                    public void run() {
//                        // TODO task to be done every 666 milliseconds
//                        // after a lapse time of 555 milliseconds
//                        timerbb = true;
//                    }
//                }, 10000, 5000);

//                else
//                {
//
//                    if(countblink/2 >= 4)
//                    {
//                        beep.start();
//
//                        //Toast.makeText(getApplicationContext(),"Hello อีดอก",Toast.LENGTH_LONG).show();
//                        //Core.putText(mRgba, "Open", new Point(mRgba.size().width/18, mRgba.size().height/5), Core.FONT_HERSHEY_SCRIPT_COMPLEX, 4, new Scalar(0,255,0),5);
//                        //Core.putText(mRgba, "Hello", new Point(mRgba.size().width/18, mRgba.size().height/5), Core.FONT_HERSHEY_SIMPLEX, 4, new Scalar(0,255,0),5);
//                        //Core.putText(onCameraFrame.this,"Hello",new Point(center.x - 50, center.y + 50), Core.FONT_HERSHEY_SIMPLEX, 4, new Scalar(0, 0, 0, 0));
//
//                    }
//                }

//            Timer timer = new Timer();
//            TimerTask t = new TimerTask() {
//                @Override
//                public void run() {
////                    if (System.currentTimeMillis() - scheduledExecutionTime() >= 10000 )
////                    Intent intent = new Intent(FdActivity.this,FdActivity.class);
////                    startActivity(intent);
////                    finish();
////                    countblink = 0;
//                    timerbb = true;
//
//
//                }
//                public void onTick() {
////                    if (System.currentTimeMillis() - scheduledExecutionTime() >= 10000 )
////                    Intent intent = new Intent(FdActivity.this,FdActivity.class);
////                    startActivity(intent);
////                    finish();
////                    countblink = 0;
////                    timerbb = true;
//
//
//                }
//                public void onFinish() {
//                    //mTextField.setText("done!");
//                    //FdActivity.this.finish();
////                    Intent intent = getIntent();
////                    intent.addFlags(Intent.FLAG_ACTIVITY_NO_ANIMATION);
////                    finish();
////                    startActivity(intent);
////                    Intent intent= new Intent(this,FdActivity.class);
////                    Intent intent = new Intent(FdActivity.this,FdActivity.class);
////                    startActivity(intent);
//                    //countblink = 0;
//
//
//                }
//            };
//            timer.scheduleAtFixedRate(t,10000,2000);

                if(!timerStarted)
                {
                    timer = Core.getTickCount();
                    timerStarted = true;
                }
                timer1 = Core.getTickCount();

                if((timer1-timer)/frequency > 60)
                {
                    countblink = 0;
                    timerStarted = false;
                }


            }


            return mRgba;

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        return true;
    }

     public void InitTimer(View v){
         flag = false;
         countblink = 0;
     }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);

        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void CreateAuxiliaryMats() {
        if (mGray.empty())
            return;

        int rows = mGray.rows();
        int cols = mGray.cols();

        if (mZoomWindow == null) {
            mZoomWindow = mRgba.submat(rows / 2 + rows / 10, rows, cols / 2
                    + cols / 10, cols);
            mZoomWindow2 = mRgba.submat(0, rows / 2 - rows / 10, cols / 2
                    + cols / 10, cols);
        }

    }

    private Rect get_template(CascadeClassifier clasificator, Rect RectAreaInterest) {
         Mat template = new Mat(); //Where is gonna be stored the eye detected data
         Mat mROI = mGray.submat(RectAreaInterest); //Matrix which contain data of the whole eye area from geometry of face
         MatOfRect eyes = new MatOfRect();
         iris = new Point();
         eye_template = new Rect();
         //detectMultiScale(const Mat& image, vector<Rect>& objects, double scaleFactor=1.1, int minNeighbors=3, int flags=0, Size minSize=Size(), Size maxSize=Size())
         clasificator.detectMultiScale(mROI, //Image which set classification. Needs to be of the type CV_8U
                 eyes, //List of rectangles where are stored possibles eyes detected
                 1.1, //Scalefactor. How much the image is reduced at each image scale
                 2,    //MinNeighbors. Specify how many neighbors each candidate rectangle should have to retain it.
                 Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE, //0 or 1.
                 new Size(10, 10), //Minimum possible object size. Objects smaller than that are ignored.
                 new Size(100,100)        //Maximum possible object size. Objects larger than that are ignored.
         );

         Rect[] eyesArray = eyes.toArray();
         for (int i = 0; i < eyesArray.length;) {
             Rect eyeDetected = eyesArray[i];
             eyeDetected.x = RectAreaInterest.x + eyeDetected.x;
             eyeDetected.y = RectAreaInterest.y + eyeDetected.y;

             mROI = mGray.submat(eyeDetected);
             mmG = Core.minMaxLoc(mROI);

             iris.x = mmG.minLoc.x + eyeDetected.x;
             iris.y = mmG.minLoc.y + eyeDetected.y;
             eye_template = new Rect((int) iris.x -  eyeDetected.width/2, (int) iris.y -  eyeDetected.height/2,  eyeDetected.width,  eyeDetected.height);

             //Imgproc.equalizeHist(template, template);
             break;
             //return template;
         }
         return eye_template;
     }

    private Rect get_template(CascadeClassifier clasificator, Rect RectAreaInterest, Size min_size, Size max_size) {
         Mat template = new Mat(); //Where is gonna be stored the eye detected data
         Mat mROI = mGray.submat(RectAreaInterest); //Matrix which contain data of the whole eye area from geometry of face
         MatOfRect eyes = new MatOfRect();
         iris = new Point();
         eye_template = new Rect();
         //detectMultiScale(const Mat& image, vector<Rect>& objects, double scaleFactor=1.1, int minNeighbors=3, int flags=0, Size minSize=Size(), Size maxSize=Size())
         clasificator.detectMultiScale(mROI, //Image which set classification. Needs to be of the type CV_8U
                 eyes, //List of rectangles where are stored possibles eyes detected
                 1.01, //Scalefactor. How much the image is reduced at each image scale
                 2,    //MinNeighbors. Specify how many neighbors each candidate rectangle should have to retain it.
                 Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE, //0 or 1.
                 min_size, //Minimum possible object size. Objects smaller than that are ignored.
                 max_size        //Maximum possible object size. Objects larger than that are ignored.
         );

         Rect[] eyesArray = eyes.toArray();
         for (int i = 0; i < eyesArray.length;) {
             Rect eyeDetected = eyesArray[i];
             eyeDetected.x = RectAreaInterest.x + eyeDetected.x;
             eyeDetected.y = RectAreaInterest.y + eyeDetected.y;

             mROI = mGray.submat(eyeDetected);
             mmG = Core.minMaxLoc(mROI);

             iris.x = mmG.minLoc.x + eyeDetected.x;
             iris.y = mmG.minLoc.y + eyeDetected.y;
             eye_template = new Rect((int) iris.x -  eyeDetected.width/2, (int) iris.y -  eyeDetected.height/2,  eyeDetected.width,  eyeDetected.height);

             //Imgproc.equalizeHist(template, template);
             break;

             //return template;
         }
         return eye_template;
     }

    private boolean match_eye(Mat mTemplate) {
         //Check for bad template size
         if (mTemplate.cols() == 0 || mTemplate.rows() == 0) {
             return false;
         }else{
             return true;
         }
     }

   /* //Matching the template
    private void match_eye(Rect area, Mat mTemplate, int type) {
        Point matchLoc;
        Mat mROI = mGray.submat(area);
        int result_cols = mROI.cols() - mTemplate.cols() + 1;
        int result_rows = mROI.rows() - mTemplate.rows() + 1;
        // Check for bad template size
        if (mTemplate.cols() == 0 || mTemplate.rows() == 0) {
            return ;
        }
        Mat mResult = new Mat(result_cols, result_rows, CvType.CV_8U);

        switch (type) {
            case TM_SQDIFF:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_SQDIFF);
                break;
            case TM_SQDIFF_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_SQDIFF_NORMED);
                break;
            case TM_CCOEFF:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCOEFF);
                break;
            case TM_CCOEFF_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCOEFF_NORMED);
                break;
            case TM_CCORR:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCORR);
                break;
            case TM_CCORR_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCORR_NORMED);
                break;
        }

        Core.MinMaxLocResult mmres = Core.minMaxLoc(mResult);
        // there is difference in matching methods - best match is max/min value
        if (type == TM_SQDIFF || type == TM_SQDIFF_NORMED) {
            matchLoc = mmres.minLoc;
        } else {
            matchLoc = mmres.maxLoc;
        }

        Point matchLoc_tx = new Point(matchLoc.x + area.x, matchLoc.y + area.y);
        Point matchLoc_ty = new Point(matchLoc.x + mTemplate.cols() + area.x,
                matchLoc.y + mTemplate.rows() + area.y);

        Imgproc.rectangle(mRgba, matchLoc_tx, matchLoc_ty, new Scalar(255, 255, 0,
                255));
        Rect rec = new Rect(matchLoc_tx,matchLoc_ty);


    }

    //Get template
    private Mat get_template(CascadeClassifier clasificator, Rect area, int size) {
        Mat template = new Mat();
        Mat mROI = mGray.submat(area);
        MatOfRect eyes = new MatOfRect();
        Point iris = new Point();
        Rect eye_template = new Rect();
        clasificator.detectMultiScale(mROI, eyes, 1.15, 2,
                Objdetect.CASCADE_FIND_BIGGEST_OBJECT
                        | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30),
                new Size());

        Rect[] eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length;) {
            Rect e = eyesArray[i];
            e.x = area.x + e.x;
            e.y = area.y + e.y;
            Rect eye_only_rectangle = new Rect((int) e.tl().x,
                    (int) (e.tl().y + e.height * 0.5), (int) e.width,
                    (int) (e.height * 0.6));
            mROI = mGray.submat(eye_only_rectangle);
            Mat vyrez = mRgba.submat(eye_only_rectangle);


            Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);

            Imgproc.circle(vyrez, mmG.minLoc, 2, new Scalar(255, 255, 255, 255), 2);
            iris.x = mmG.minLoc.x + eye_only_rectangle.x;
            iris.y = mmG.minLoc.y + eye_only_rectangle.y;
            eye_template = new Rect((int) iris.x - size / 2, (int) iris.y
                    - size / 2, size, size);
            //Imgproc.rectangle(mRgba, eye_template.tl(), eye_template.br(), new Scalar(255, 0, 0, 255), 2);
            template = (mGray.submat(eye_template)).clone();
            return template;
        }
        return template;
    }

    public void onRecreateClick(View v)
    {
        learn_frames = 0;
    }*/

}
