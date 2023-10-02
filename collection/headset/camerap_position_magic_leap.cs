using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Text;
using UnityEngine.XR.MagicLeap;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System;

public class camera_position : MonoBehaviour
{
    private StreamWriter writer;
    private string path;
    public Transform tf;
    public Vector3 position;
    //public Quaternion rotation;
    public Vector3 rotation;
    public Vector3 rotation_radious;
    public GameObject a;
    public int ran;
    public bool finish = false;
    public int VideoID;
    public string VideoName;
    public double TimeFrame;

    private MLInput.Controller _controller;

    // For udp client use
    private static int localPort;

    // prefs
    private string IP; // define in init
    private int port; // define in init
    
    // "connection" things
    IPEndPoint remoteEndPoint;
    UdpClient client;

    // gui
    string strMessage = "";

    // For udp receiver use
    UdpClient UDPReceiver;
    int recvPort;


    // Use this for initialization
    void Start()
    {
        // init udp client
        init();

        // init udp receiver
        // recvInit();

        //path = "Data/" + "log_pos.txt";
        //writer = new StreamWriter(path, true);
        tf = this.transform;
        finish = false;
        //path = System.IO.Path.Combine(Application.persistentDataPath, Config.UserID + "-" + VideoName + "-Log" + UnityEngine.Random.Range(1, 10000) + ".txt");
        path = System.IO.Path.Combine(Application.persistentDataPath, "-Log" + UnityEngine.Random.Range(1, 10000) + ".txt");
        //path = System.IO.Path.Combine(Application.persistentDataPath, "log_test1.txt");
        writer = new StreamWriter(path, true);
        TimeFrame = 0;
        Config.Log_Times = 0;
        //path = Application.dataPath + "Assets/Logs/log4.txt";

        MLInput.Start();
        _controller = MLInput.GetController(MLInput.Hand.Left);
    }

    // Update is called once per frame
    void Update()
    {
        TimeFrame += Time.deltaTime;
        if (TimeFrame < Config.Log_Times * Config.Log_sleepInterval * 0.001f) return;
        if (!finish)
        {
            LogViewport();
        }
        Config.Log_Times++;
        Config.Playing_Frame++;
        // finish = Config.ReadFinish;
        if (finish)
        {
            Destroy(a);
            Config.vFinish[VideoID] = true;
            // sendString("aaa");
            // SceneManager.LoadScene("Assets/MagicLeap_Chose.unity");
        }

        CheckTrigger();
    }

    void OnDestroy()
    {
        writer.Flush();
        writer.Close();
    }

    void LogViewport()
    {
        position = tf.position;
        rotation = tf.rotation.eulerAngles;
        rotation_radious = rotation * Mathf.Deg2Rad;
        double now = getUnixTimestamp();
        // WriteString(Config.Playing_Frame, position, rotation, rotation_radious, TimeFrame);
        WriteString(Config.Playing_Frame, position, rotation, rotation_radious, now);
    }

    void WriteString(int frameID, Vector3 input, Vector3 input2, Vector3 input3, double Time)
    {
        StringBuilder sb = new StringBuilder();
        sb.Append(frameID);
        sb.Append(",");
        sb.Append(input.x.ToString("F4"));
        sb.Append(",");
        sb.Append(input.y.ToString("F4"));
        sb.Append(",");
        sb.Append(input.z.ToString("F4"));
        sb.Append(",");
        sb.Append(input2.x.ToString("F4"));
        sb.Append(",");
        sb.Append(input2.y.ToString("F4"));
        sb.Append(",");
        sb.Append(input2.z.ToString("F4"));
        sb.Append(",");
        sb.Append(input3.x.ToString("F4"));
        sb.Append(",");
        sb.Append(input3.y.ToString("F4"));
        sb.Append(",");
        sb.Append(input3.z.ToString("F4"));
        sb.Append(",");
        sb.Append(Time.ToString("F4"));
        writer.WriteLine(sb.ToString());

        // double t1 = getUnixTimestamp();
        // send data
        sendString(sb.ToString());
        // try to receive data
        // recvData();
        // double t2 = getUnixTimestamp();
        // writer.WriteLine((t2 - t1).ToString);
    }

    void CheckTrigger()
    {
        if (_controller.TriggerValue > 0.2f)
        {
            finish = true;
        }
    }

    // init
    public void init()
    {
        // Endpoint definition
        // print("UDP Client Init()");

        // define
        // IP = "10.0.0.191";
        // IP = "192.168.0.123";
        IP = "192.168.0.131";
        port = 5050;

        // --------------------
        // Send
        // --------------------
        remoteEndPoint = new IPEndPoint(IPAddress.Parse(IP), port);
        client = new UdpClient();

        // status
        // print("Sending to " + IP + " : " + port);
        // print("Testing: nc -lu " + IP + " : " + port);
    }

    private void sendString(string message)
    {
        try
        {
            byte[] data = Encoding.UTF8.GetBytes(message);
            //double t1 = getUnixTimestamp();
            client.Send(data, data.Length, remoteEndPoint);

            //byte[] rcvData = client.Receive(ref remoteEndPoint);
            //double t2 = getUnixTimestamp();
            //writer.WriteLine((t2 - t1).ToString());
        }
        catch (Exception err)
        {
            print(err.ToString());
        }
    }

    private double getUnixTimestamp()
    {
        // DateTime baseTime = TimeZone.CurrentTimeZone.ToLocalTime(new DateTime(1970, 1, 1, 0, 0, 0));
        // DateTime now = DateTime.Now.AddHours(-1.0);
        // double result = (now - baseTime).TotalMilliseconds / 1000.0;
        double result = (DateTime.Now.ToUniversalTime().Ticks - 621355968000000000) / 10000000.0;
        return result;
    }

    // udp receiver init
    /*
    private void recvInit()
    {
        recvPort = 5050;
        UDPReceiver = new UdpClient(recvPort);
    }
    */

    // udp receiver receives data
    /*
    private void recvData()
    {
        try
        {
            IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
            byte[] data = UDPReceiver.Receive(ref anyIP);
        }
        catch(Exception err)
        {
            print(err.ToString());
        }
    }
    */
}
