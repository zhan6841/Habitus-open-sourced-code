//
// Created by anlan.
//

#include <stdio.h>
#include <pcap.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netinet/tcp.h>
#include <sys/time.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <net/if.h>

#define NICNAME "wigig0"
#define NICNAME2 "wlan0"

typedef unsigned char BYTE;
typedef unsigned int DWORD;
typedef unsigned short WORD;

/* default snap length (maximum bytes per packet to capture) */
// #define SNAP_LEN 1518
#define SNAP_LEN 96

/* ethernet headers are always exactly 14 bytes [1] */
#define SIZE_ETHERNET 14

/* Ethernet addresses are 6 bytes */
#define ETHER_ADDR_LEN	6

/* Ethernet header */
struct sniff_ethernet {
    u_char  ether_dhost[ETHER_ADDR_LEN];    /* destination host address */
    u_char  ether_shost[ETHER_ADDR_LEN];    /* source host address */
    u_short ether_type;                     /* IP? ARP? RARP? etc */
};

/* IP header */
struct sniff_ip {
    u_char  ip_vhl;                 /* version << 4 | header length >> 2 */
    u_char  ip_tos;                 /* type of service */
    u_short ip_len;                 /* total length */
    u_short ip_id;                  /* identification */
    u_short ip_off;                 /* fragment offset field */
#define IP_RF 0x8000            /* reserved fragment flag */
#define IP_DF 0x4000            /* don't fragment flag */
#define IP_MF 0x2000            /* more fragments flag */
#define IP_OFFMASK 0x1fff       /* mask for fragmenting bits */
    u_char  ip_ttl;                 /* time to live */
    u_char  ip_p;                   /* protocol */
    u_short ip_sum;                 /* checksum */
    struct  in_addr ip_src,ip_dst;  /* source and dest address */
};
#define IP_HL(ip)               (((ip)->ip_vhl) & 0x0f)
#define IP_V(ip)                (((ip)->ip_vhl) >> 4)

/* TCP header */
typedef u_int tcp_seq;

struct sniff_tcp {
    u_short th_sport;               /* source port */
    u_short th_dport;               /* destination port */
    tcp_seq th_seq;                 /* sequence number */
    tcp_seq th_ack;                 /* acknowledgement number */
    u_char  th_offx2;               /* data offset, rsvd */
#define TH_OFF(th)      (((th)->th_offx2 & 0xf0) >> 4)
    u_char  th_flags;
#define TH_FIN  0x01
#define TH_SYN  0x02
#define TH_RST  0x04
#define TH_PUSH 0x08
#define TH_ACK  0x10
#define TH_URG  0x20
#define TH_ECE  0x40
#define TH_CWR  0x80
#define TH_FLAGS        (TH_FIN|TH_SYN|TH_RST|TH_ACK|TH_URG|TH_ECE|TH_CWR)
    u_short th_win;                 /* window */
    u_short th_sum;                 /* checksum */
    u_short th_urp;                 /* urgent pointer */
};

#define MAX_SIZE 1000000
#define LOOK_AHEAD_WIN 2 // 2 seconds

// nicID
#define NICAD 0
#define NICAC 1

// for ad use
pcap_t *adHandle;
pcap_dumper_t *adPktFp; // file for dumping all the packets
double firstADPktTS = 0.0;
double adTimestamps[MAX_SIZE];
int adBytes[MAX_SIZE];
int adPtr = 0;
int adSize = 0;
// for ac use
pcap_t *acHandle;
pcap_dumper_t *acPktFp; // file for dumping all the packets
double firstACPktTS = 0.0;
double acTimestamps[MAX_SIZE];
int acBytes[MAX_SIZE];
int acPtr = 0;
int acSize = 0;

// for Magic Leap Receiver use
char *svrIP = NULL;
int svrPort = 5050;
double targetDuration = 120.0;

// for ad signal strength
double adSS = -1000.0;

// for ac signal strength
double acSS = -1000.0;

// for panorama camera information
int frameID = 0;
// for 2D camera infomation
int svrPort2 = 5051;

// for filtering
int filterADPort = 5051;
int filterACPort = 5052;

// for data collection
char* infoFile = NULL;
char* traceFile = NULL;

void MyAssert(int x, int assertID);
double NDKGetTime();

double ComputeThroughput(int nicID, double currentTS, double win);

void *CaptureADPktsThread(void *args);
void got_packet_ad(u_char *args, const struct pcap_pkthdr *header, const u_char *packet);

void *CaptureACPktsThread(void *args);
void got_packet_ac(u_char *args, const struct pcap_pkthdr *header, const u_char *packet);

void *MLReceiverUDPThread(void *args);

void *DumpADSignalStrengthThread(void *args);

void *DumpACSignalStrengthThread(void *args);

void *CameraReceiverUDPThread(void *args); // to get the updated android camera frame id

int main(int argc, char **argv){

    if (argc != 9) {
        printf("Usage: %s [svrIP] [svrPort] [svrPort2] [targetDuration] [filterADPort] [filterACPort] [infoFile] [traceFile]\n", argv[0]);
        return -1;
    }
    svrIP = argv[1];
    svrPort = atoi(argv[2]);
    svrPort2 = atoi(argv[3]);
    targetDuration = atof(argv[4]);
    filterADPort = atoi(argv[5]);
    filterACPort = atoi(argv[6]);
    infoFile = argv[7];
    traceFile = argv[8];

    pthread_t cameraReceiverUDPThreadTid;
    pthread_t dumpADSignalStrengthThreadTid, dumpACSignalStrengthThreadTid;
    pthread_t captureADPktsThreadTid, captureACPktsThreadTid;
    pthread_t mlReceiverUDPThreadTid;

    pthread_create(&cameraReceiverUDPThreadTid, NULL, CameraReceiverUDPThread, NULL);
    pthread_create(&dumpADSignalStrengthThreadTid, NULL, DumpADSignalStrengthThread, NULL);
    pthread_create(&dumpACSignalStrengthThreadTid, NULL, DumpACSignalStrengthThread, NULL);
    pthread_create(&captureADPktsThreadTid, NULL, CaptureADPktsThread, NULL);
    pthread_create(&captureACPktsThreadTid, NULL, CaptureACPktsThread, NULL);

    // wait for camera
    while(1){
        printf("Main FrameID: %d\n", frameID);
        if(frameID > 0){
            // printf("Main FrameID: %d\n", frameID);
            break;
        }
        sleep(1);
    }
    pthread_create(&mlReceiverUDPThreadTid, NULL, MLReceiverUDPThread, NULL);

    pthread_join(captureADPktsThreadTid, NULL);
    pthread_join(captureACPktsThreadTid, NULL);
    pthread_join(mlReceiverUDPThreadTid, NULL);

    // pthread_cancel(dumpADSignalStrengthThreadTid);
    // pthread_cancel(dumpACSignalStrengthThreadTid);

    // pthread_join(dumpADSignalStrengthThreadTid, NULL);
    // pthread_join(dumpACSignalStrengthThreadTid, NULL);

    // log to info.txt
    FILE *infoFp = fopen(infoFile, "w");
    double firstPktTS = (firstADPktTS < firstACPktTS) ? firstADPktTS : firstACPktTS;
    fprintf(infoFp, "%.1f\n%.1f\n%.6f", (double)LOOK_AHEAD_WIN, targetDuration, firstPktTS);
    fclose(infoFp);

    return 0;
}

void *CaptureADPktsThread(void *args){
    printf("AD Collector Started src port %d.\n", filterADPort);

    char *dev = NICNAME;			/* capture device name */
    char errbuf[PCAP_ERRBUF_SIZE];		/* error buffer */
    pcap_t *handle;				/* packet capture handle */

    char filter_exp[64];
    sprintf(filter_exp, "tcp src port %d", filterADPort);
    // char filter_exp[] = "tcp src port 5050";		/* filter expression [3] */
    struct bpf_program fp;			/* compiled filter program (expression) */
    bpf_u_int32 mask;			/* subnet mask */
    bpf_u_int32 net;			/* ip */
    int num_packets = 0;			/* number of packets to capture */

    /* get network number and mask associated with capture device */
    if (pcap_lookupnet(dev, &net, &mask, errbuf) == -1) {
        fprintf(stderr, "Couldn't get netmask for device %s: %s\n",
                dev, errbuf);
        net = 0;
        mask = 0;
    }

    /* print capture info */
    printf("Device: %s\n", dev);
    printf("Number of packets: %d\n", num_packets);
    printf("Filter expression: %s\n", filter_exp);

    /* open capture device */
    handle = pcap_open_live(dev, SNAP_LEN, 0, 1000, errbuf);
    if (handle == NULL) {
        fprintf(stderr, "Couldn't open device %s: %s\n", dev, errbuf);
        exit(EXIT_FAILURE);
    }
    // in order to use pcap_breakloop in other threads
    adHandle = handle;

    /* make sure we're capturing on an Ethernet device [2] */
    if (pcap_datalink(handle) != DLT_EN10MB) {
        fprintf(stderr, "%s is not an Ethernet\n", dev);
        exit(EXIT_FAILURE);
    }

    /* compile the filter expression */
    if (pcap_compile(handle, &fp, filter_exp, 0, net) == -1) {
        fprintf(stderr, "Couldn't parse filter %s: %s\n",
                filter_exp, pcap_geterr(handle));
        exit(EXIT_FAILURE);
    }

    /* apply the compiled filter */
    if (pcap_setfilter(handle, &fp) == -1) {
        fprintf(stderr, "Couldn't install filter %s: %s\n",
                filter_exp, pcap_geterr(handle));
        exit(EXIT_FAILURE);
    }

    // open .pcap file for dumping
//    adPktFp = pcap_dump_open(handle, "ad.pcap");

    /* now we can set our callback function */
    pcap_loop(handle, num_packets, got_packet_ad, NULL);

    /* cleanup */
//    pcap_dump_flush(adPktFp);
//    pcap_dump_close(adPktFp);
    pcap_freecode(&fp);
    pcap_close(handle);

    printf("AD Capture complete.\n");

    return NULL;
}

/*
 * dissect/print packet
 */
void got_packet_ad(u_char *args, const struct pcap_pkthdr *header, const u_char *packet){

    static int count = 1;                   /* packet counter */

    /* declare pointers to packet headers */
    const struct sniff_ethernet *ethernet;  /* The ethernet header [1] */
    const struct sniff_ip *ip;              /* The IP header */
    const struct sniff_tcp *tcp;            /* The TCP header */
    const char *payload;                    /* Packet payload */

    int size_ip;
    int size_tcp;
    int size_payload;

    // printf("AD Packet number %d:\n", count);
    count++;

    // printf("Capture Size: %d", header->caplen);

    /* define ethernet header */
    ethernet = (struct sniff_ethernet*)(packet);

    /* define/compute ip header offset */
    ip = (struct sniff_ip*)(packet + SIZE_ETHERNET);
    size_ip = IP_HL(ip)*4;
    if (size_ip < 20) {
        printf("   * Invalid IP header length: %u bytes\n", size_ip);
        return;
    }

    /* print source and destination IP addresses */
    // printf("       From: %s\n", inet_ntoa(ip->ip_src));
    // printf("         To: %s\n", inet_ntoa(ip->ip_dst));

    /* determine protocol */
    switch(ip->ip_p) {
        case IPPROTO_TCP:
            // printf("   Protocol: TCP\n");
            break;
        case IPPROTO_UDP:
            // printf("   Protocol: UDP\n");
            return;
        case IPPROTO_ICMP:
            // printf("   Protocol: ICMP\n");
            return;
        case IPPROTO_IP:
            // printf("   Protocol: IP\n");
            return;
        default:
            // printf("   Protocol: unknown\n");
            return;
    }

    /*
     *  OK, this packet is TCP.
     */

    /* define/compute tcp header offset */
    tcp = (struct sniff_tcp*)(packet + SIZE_ETHERNET + size_ip);
    size_tcp = TH_OFF(tcp)*4;
    if (size_tcp < 20) {
        printf("   * Invalid TCP header length: %u bytes\n", size_tcp);
        return;
    }

    // printf("   Src port: %d\n", ntohs(tcp->th_sport));
    // printf("   Dst port: %d\n", ntohs(tcp->th_dport));

    /* define/compute tcp payload (segment) offset */
//    payload = (u_char *)(packet + SIZE_ETHERNET + size_ip + size_tcp);

    /* compute tcp payload (segment) size */
    size_payload = ntohs(ip->ip_len) - (size_ip + size_tcp);

    /*
     * Print payload data; it might be binary, so don't just
     * treat it as a string.
     */
    // if (size_payload > 0) {
    // printf("   Payload (%d bytes):\n", size_payload);
    // print_payload(payload, size_payload);
    // }

    if(ntohs(tcp->th_sport) == filterADPort){
        double t = (double)(header->ts.tv_sec + (double)header->ts.tv_usec / 1e6f);
        adTimestamps[adPtr] = t;
        adBytes[adPtr] = size_payload;
        adPtr = (adPtr + 1) % MAX_SIZE;
        adSize += 1;
    }

    // dump packet
    // pcap_dump(adPktFp, header, packet);

    if(firstADPktTS == 0.0){
        firstADPktTS = (double)(header->ts.tv_sec + (double)header->ts.tv_usec / 1e6f);
    }

    return;
}

void *CaptureACPktsThread(void *args){
    printf("AC Collector Started src port %d.\n", filterACPort);

    char *dev = NICNAME2;			/* capture device name */
    char errbuf[PCAP_ERRBUF_SIZE];		/* error buffer */
    pcap_t *handle;				/* packet capture handle */

    char filter_exp[64];
    sprintf(filter_exp, "tcp src port %d", filterACPort);
//    char filter_exp[] = "tcp src port 5051";		/* filter expression [3] */
    struct bpf_program fp;			/* compiled filter program (expression) */
    bpf_u_int32 mask;			/* subnet mask */
    bpf_u_int32 net;			/* ip */
    int num_packets = 0;			/* number of packets to capture */

    /* get network number and mask associated with capture device */
    if (pcap_lookupnet(dev, &net, &mask, errbuf) == -1) {
        fprintf(stderr, "Couldn't get netmask for device %s: %s\n",
                dev, errbuf);
        net = 0;
        mask = 0;
    }

    /* print capture info */
    printf("Device: %s\n", dev);
    printf("Number of packets: %d\n", num_packets);
    printf("Filter expression: %s\n", filter_exp);

    /* open capture device */
    handle = pcap_open_live(dev, SNAP_LEN, 0, 1000, errbuf);
    if (handle == NULL) {
        fprintf(stderr, "Couldn't open device %s: %s\n", dev, errbuf);
        exit(EXIT_FAILURE);
    }
    // in order to use pcap_breakloop in other threads
    acHandle = handle;

    /* make sure we're capturing on an Ethernet device [2] */
    if (pcap_datalink(handle) != DLT_EN10MB) {
        fprintf(stderr, "%s is not an Ethernet\n", dev);
        exit(EXIT_FAILURE);
    }

    /* compile the filter expression */
    if (pcap_compile(handle, &fp, filter_exp, 0, net) == -1) {
        fprintf(stderr, "Couldn't parse filter %s: %s\n",
                filter_exp, pcap_geterr(handle));
        exit(EXIT_FAILURE);
    }

    /* apply the compiled filter */
    if (pcap_setfilter(handle, &fp) == -1) {
        fprintf(stderr, "Couldn't install filter %s: %s\n",
                filter_exp, pcap_geterr(handle));
        exit(EXIT_FAILURE);
    }

    // open .pcap file for dumping
//    acPktFp = pcap_dump_open(handle, "ac.pcap");

    /* now we can set our callback function */
    pcap_loop(handle, num_packets, got_packet_ac, NULL);

    /* cleanup */
//    pcap_dump_flush(acPktFp);
//    pcap_dump_close(acPktFp);
    pcap_freecode(&fp);
    pcap_close(handle);

    printf("AC Capture complete.\n");

    return NULL;
}

/*
 * dissect/print packet
 */
void got_packet_ac(u_char *args, const struct pcap_pkthdr *header, const u_char *packet){

    static int count = 1;                   /* packet counter */

    /* declare pointers to packet headers */
    const struct sniff_ethernet *ethernet;  /* The ethernet header [1] */
    const struct sniff_ip *ip;              /* The IP header */
    const struct sniff_tcp *tcp;            /* The TCP header */
    const char *payload;                    /* Packet payload */

    int size_ip;
    int size_tcp;
    int size_payload;

    // printf("AC Packet number %d:\n", count);
    count++;

    // printf("Capture Size: %d", header->caplen);

    /* define ethernet header */
    ethernet = (struct sniff_ethernet*)(packet);

    /* define/compute ip header offset */
    ip = (struct sniff_ip*)(packet + SIZE_ETHERNET);
    size_ip = IP_HL(ip)*4;
    if (size_ip < 20) {
        printf("   * Invalid IP header length: %u bytes\n", size_ip);
        return;
    }

    /* print source and destination IP addresses */
    // printf("       From: %s\n", inet_ntoa(ip->ip_src));
    // printf("         To: %s\n", inet_ntoa(ip->ip_dst));

    /* determine protocol */
    switch(ip->ip_p) {
        case IPPROTO_TCP:
            // printf("   Protocol: TCP\n");
            break;
        case IPPROTO_UDP:
            // printf("   Protocol: UDP\n");
            return;
        case IPPROTO_ICMP:
            // printf("   Protocol: ICMP\n");
            return;
        case IPPROTO_IP:
            // printf("   Protocol: IP\n");
            return;
        default:
            // printf("   Protocol: unknown\n");
            return;
    }

    /*
     *  OK, this packet is TCP.
     */

    /* define/compute tcp header offset */
    tcp = (struct sniff_tcp*)(packet + SIZE_ETHERNET + size_ip);
    size_tcp = TH_OFF(tcp)*4;
    if (size_tcp < 20) {
        printf("   * Invalid TCP header length: %u bytes\n", size_tcp);
        return;
    }

    // printf("   Src port: %d\n", ntohs(tcp->th_sport));
    // printf("   Dst port: %d\n", ntohs(tcp->th_dport));

    /* define/compute tcp payload (segment) offset */
//    payload = (u_char *)(packet + SIZE_ETHERNET + size_ip + size_tcp);

    /* compute tcp payload (segment) size */
    size_payload = ntohs(ip->ip_len) - (size_ip + size_tcp);

    /*
     * Print payload data; it might be binary, so don't just
     * treat it as a string.
     */
    // if (size_payload > 0) {
    // printf("   Payload (%d bytes):\n", size_payload);
    // print_payload(payload, size_payload);
    // }

    if(ntohs(tcp->th_sport) == filterACPort){
        double t = (double)(header->ts.tv_sec + (double)header->ts.tv_usec / 1e6f);
        acTimestamps[acPtr] = t;
        acBytes[acPtr] = size_payload;
        acPtr = (acPtr + 1) % MAX_SIZE;
        acSize += 1;
    }

    // dump packet
//    pcap_dump(acPktFp, header, packet);

    if(firstACPktTS == 0.0){
        firstACPktTS = (double)(header->ts.tv_sec + (double)header->ts.tv_usec / 1e6f);
    }

    return;
}

// nicID == 0: ad, nicID == 1: ac
double ComputeThroughput(int nicID, double currentTS, double win){

    double throughput = 0.0;
    double totalBytes = 0.0;
    double duration = win;
    int end;
    int curSize;
    int idx;
    int pktCounts;

    // double t1 = NDKGetTime();
    // double t2;

    if(nicID == 0){
        // ad throughput
        end = adSize > MAX_SIZE ? adSize % MAX_SIZE : 0;
        curSize = adSize - 1;
        idx = curSize % MAX_SIZE;
        pktCounts = 0;
        while(1){

            if(adTimestamps[idx] > currentTS){
                curSize -= 1;
                idx = curSize % MAX_SIZE;
                continue;
            }

            if(adTimestamps[idx] < currentTS - win){
                break;
            }

            totalBytes += adBytes[idx];
            pktCounts += 1;

            if(idx == end){
                break;
            }

            curSize -= 1;
            idx = curSize % MAX_SIZE;
        }
        throughput = totalBytes * 8.0 / duration;

        // t2 = NDKGetTime();
        // printf("AD throughput %.6f Mbps, pkt counts %d, time %.6f\n", throughput/(1000.0 * 1000.0), pktCounts, (t2 - t1));
    }
    else if(nicID == 1){
        // ac throughput
        end = acSize > MAX_SIZE ? acSize % MAX_SIZE : 0;
        curSize = acSize - 1;
        idx = curSize % MAX_SIZE;
        pktCounts = 0;
        while(1){

            if(acTimestamps[idx] > currentTS){
                curSize -= 1;
                idx = curSize % MAX_SIZE;
                continue;
            }

            if(acTimestamps[idx] < currentTS - win){
                break;
            }

            totalBytes += acBytes[idx];
            pktCounts += 1;

            if(idx == end){
                break;
            }

            curSize -= 1;
            idx = curSize % MAX_SIZE;
        }
        throughput = totalBytes * 8.0 / duration;

        // t2 = NDKGetTime();
        // printf("AD throughput %.6f Mbps, pkt counts %d, time %.6f\n", throughput/(1000.0 * 1000.0), pktCounts, (t2 - t1));
    }

    return throughput;
}

void *MLReceiverUDPThread(void *args){
    printf("Magic Leap Receiver UDP Started.\n");

    int sockFD = socket(AF_INET, SOCK_DGRAM, 0);
    MyAssert(sockFD >= 0, 1778);

    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(struct sockaddr_in));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons((unsigned short) svrPort);
    inet_pton(AF_INET, svrIP, &serverAddr.sin_addr);

    int optval = 1;
    int r = setsockopt(sockFD, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
    MyAssert(r == 0, 1762);

    if (bind(sockFD, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) != 0) {
        MyAssert(0, 1779);
    }

    int nConn = 0;
    int i;

    static const int BUF_SIZE = 1*1024*1024;
    BYTE * buf = (BYTE*)malloc(BUF_SIZE);
    MyAssert(buf != NULL, 1784);

    memset(buf, 0, BUF_SIZE);

    FILE *fp;
    fp = fopen(traceFile, "w");

    int n;
    double tStart;
    // double tStartReal;
    double adSS_;
    double acSS_;
    while(1) {
        struct sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);

        printf("Conn=%d\n", ++nConn);
        // nConn += 1;
        // ++nConn;

        n = recvfrom(sockFD, buf, BUF_SIZE, 0, (struct sockaddr *)&clientAddr, &clientAddrLen);
        MyAssert(n >= 0, 1782);
        // printf("Received %d Bytes.\n", n);

        // // get the real timestamp
        // char tsChar[20];
        // int count = 0;
        // for(int i = n - 1; i >= 0; i--){
        // 	count += 1;
        // 	if(buf[i] == '.'){
        // 		break;
        // 	}
        // }
        // // printf("count=%d\n", count);
        // int startIdx = n-(10+count);
        // for(int i = startIdx; i < n; i++){
        // 	tsChar[i-startIdx] = buf[i];
        // }
        // tsChar[10+count] = '\0';
        // double realTS = atof(tsChar);
        // // printf("%.6f,%.6f,%.6f\n", realTS, currentTS, currentTS-realTS);

        if(nConn == 1){
            tStart = NDKGetTime();
            // tStartReal = realTS;
        }

        // if(buf[0] == 'a' && buf[1] == 'a' && buf[2] == 'a'){
        // 	break;
        // }

        double currentTS = NDKGetTime();

        double adthroughput = ComputeThroughput(NICAD, currentTS, LOOK_AHEAD_WIN);
        double acthroughput = ComputeThroughput(NICAC, currentTS, LOOK_AHEAD_WIN);
        // double adthroughput = ComputeThroughput(NICAD, realTS, LOOK_AHEAD_WIN);
        // double acthroughput = ComputeThroughput(NICAC, realTS, LOOK_AHEAD_WIN);

        adSS_ = adSS;
        acSS_ = acSS;

        // 6DoF, ad throughput, ac throughput, timestamp
        fprintf(fp, "%s,%d,%.6f,%.6f,%d,%d,%.6f\n", buf, frameID, adSS_, acSS_, (int)adthroughput, (int)acthroughput, currentTS);

        printf("%.6f,%.6f\n", adthroughput / 1024.0 / 1024.0, acthroughput / 1024.0 / 1024.0);

        // printf("%s\n", buf);
        // for(int i = 0; i < n; i++){
        //     printf("%c", buf[i]);
        // }
        // printf("\n");

        if(currentTS - tStart > targetDuration){
            break;
        }
        // if(realTS - tStartReal > targetDuration){
        // 	break;
        // }
    }

    fclose(fp);
    close(sockFD);

    // terminate pcap loops
    pcap_breakloop(adHandle);
    pcap_breakloop(acHandle);

    return NULL;
}

void *DumpADSignalStrengthThread(void *args) {

    FILE *ssFp;

    while(1){
        ssFp = popen("iw dev wigig0 link | grep 'signal' | awk '{print $2}'", "r");

        if (ssFp == NULL) {
            printf("Failed to run command on wigig0\n" );
            exit(1);
        }

        char result[10];

        if(fgets(result, sizeof(result), ssFp) != NULL){
            // printf("%s", result);
            adSS = atof(result);
        }
        else{
            adSS = -1000.0;
        }

        pclose(ssFp);
    }

    return NULL;
}

void *DumpACSignalStrengthThread(void *args) {

    FILE *ssFp;

    while(1){
        ssFp = popen("iw dev wlan0 link | grep 'signal' | awk '{print $2}'", "r");

        if (ssFp == NULL) {
            printf("Failed to run command on wlan0\n" );
            exit(1);
        }

        char result[10];

        if(fgets(result, sizeof(result), ssFp) != NULL){
            // printf("%s", result);
            acSS = atof(result);
        }
        else{
            acSS = -1000.0;
        }

        pclose(ssFp);
    }

    return NULL;
}

void MyAssert(int x, int assertID) {
    if (!x) {
        fprintf(stderr, "Assertion failure: %d\n", assertID);
        fprintf(stderr, "errno = %d (%s)\n", errno, strerror(errno));
        exit(-1);
    }
}

double NDKGetTime() {
    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    double t = res.tv_sec + (double) res.tv_nsec / 1e9f;
    return t;
}

void *CameraReceiverUDPThread(void *args){
    printf("Camera Receiver UDP Started.\n");

    int sockFD = socket(AF_INET, SOCK_DGRAM, 0);
    MyAssert(sockFD >= 0, 1128);

    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(struct sockaddr_in));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons((unsigned short) svrPort2);
    inet_pton(AF_INET, svrIP, &serverAddr.sin_addr);

    int optval = 1;
    int r = setsockopt(sockFD, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
    MyAssert(r == 0, 1129);

    if (bind(sockFD, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) != 0) {
        MyAssert(0, 1130);
    }

    // int nConn = 0;

    static const int BUF_SIZE = 1*1024;
    BYTE * buf = (BYTE*)malloc(BUF_SIZE);
    MyAssert(buf != NULL, 1131);

    memset(buf, 0, BUF_SIZE);

    int n;
    while(1) {
        struct sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);

        // printf("Conn=%d\n", ++nConn);

        n = recvfrom(sockFD, buf, BUF_SIZE, 0, (struct sockaddr *)&clientAddr, &clientAddrLen);
        // printf("Received %d Bytes.\n", n);
        MyAssert(n == 4, 1132);

        frameID = ((int *)buf)[0];
        // frameID += 1;
        // printf("frameID=%d\n", frameID);
        // printf("%s\n", buf);
        // printf("%ld\n", ((long *)buf)[0]);
    }

    close(sockFD);

    return NULL;
}