using UnityEngine;
using System.Collections;
using System.IO;

public class CameraMove : MonoBehaviour {
	public int corridor = 1;
	public int pass = 1;
	public float frameRate = 60;			//frames per second
	public float movementSpeed = 1.4f;		//meters per second


	public float delay = 1;

	int stepN = 0;
	FileStream file;
	StreamWriter sw;
	bool ended = false;
	float ax = 0;
	float ay = 0;

	float yheight = 1.8f;
	float dpy = 0;
	float dpz = 0;
	float dax = 0;
	float day = 0;

	void Start(){
		file = new FileStream("ground_truth_C"+corridor.ToString()+"_P"+pass.ToString()+".csv", FileMode.Create, FileAccess.Write);
		sw = new StreamWriter(file);
	}

	void Update(){
		delay -= Time.deltaTime;
		if (delay>0)	return;

		if (transform.position.x<50){
			NextStep();
		}else{
			if (!ended){
				End();
				ended = true;
			}
		}
	}

    void NextStep()
    {
        Vector3 pos = transform.position;

        //take image
        string filename = "image_C" + corridor.ToString() + "_P" + pass.ToString() + "_S" + stepN.ToString() + ".png";
        ScreenCapture.CaptureScreenshot(filename);

        //write position
        sw.WriteLine(filename + ","
                     + pos.x.ToString() + ","
                     + pos.y.ToString() + ","
                     + pos.z.ToString() + ","
                     + ax.ToString() + ","
                     + ay.ToString());
        file.Flush();
        stepN++;
    }
	void End(){
		sw.Close();
	}
}