using UnityEngine;
using TMPro;
using System.Net.Sockets;
using System.Net;
using System.Text;
using System;

#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.XR.Management;
#else
using UnityEngine.XR.Management;
#endif

namespace Unity.Template.VR
{
    internal class XRPlatformControllerSetup : MonoBehaviour
    {
        [SerializeField]
        GameObject m_LeftController;

        [SerializeField]
        GameObject m_RightController;
        
        [SerializeField]
        GameObject m_LeftControllerOculusPackage;

        [SerializeField]
        GameObject m_RightControllerOculusPackage;

        public GameObject cameraObj;
        public TextMeshProUGUI textCoord;

        private readonly int localPort = 11456;
        //private readonly string remoteIP = "10.131.24.204";
        //private readonly int remotePort = 22455;
        //private readonly string remoteIP = "10.131.73.21";
        private readonly string remoteIP = "192.168.0.123";
        private readonly int remotePort = 5050;

        private UdpClient udpClient;

        void Start()
        {
#if UNITY_EDITOR
            var loaders = XRGeneralSettingsPerBuildTarget.XRGeneralSettingsForBuildTarget(BuildTargetGroup.Standalone).Manager.activeLoaders;
#else
            var loaders = XRGeneralSettings.Instance.Manager.activeLoaders;
#endif
            
            foreach (var loader in loaders)
            {
                if (loader.name.Equals("Oculus Loader"))
                {
                    m_RightController.SetActive(false);
                    m_LeftController.SetActive(false);
                    m_RightControllerOculusPackage.SetActive(true);
                    m_LeftControllerOculusPackage.SetActive(true);
                }
            }

            try
            {
                udpClient = new UdpClient(localPort);
                udpClient.Connect(remoteIP, remotePort);
            }
            catch (Exception e)
            {
                Debug.Log("Exception thrown " + e.Message);
            }
        }

        // Update is called once per frame
        void Update()
        {
            Vector3 pos = cameraObj.transform.position;
            Vector3 ang = cameraObj.transform.eulerAngles;
            //textCoord.text = cameraObj.transform.position.ToString() + ",";
            //textCoord.text += cameraObj.transform.eulerAngles.ToString();

            textCoord.text = string.Format("{0},{1},{2},{3},{4},{5}", pos[0], pos[1], pos[2], ang[0], ang[1], ang[2]);
            //Debug.Log(textCoord.text);

            SendUdpMessage(textCoord.text);
        }


        void SendUdpMessage(string message)
        {
            try
            {
                byte[] sendBytes = Encoding.UTF8.GetBytes(message);
                udpClient.Send(sendBytes, sendBytes.Length);
                //udpClient.Send(sendBytes, sendBytes.Length, remoteIP, remotePort);
            }
            catch (Exception e)
            {
                Debug.Log("Exception thrown " + e.Message);
            }
        }
    }
}