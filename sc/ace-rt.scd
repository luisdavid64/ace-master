//
// Hear-through system for real-time Auditory Contrast Enhancement (ACE)
// SuperCollider code
// 2019 Marian Weger
//


// 1. Install ACE Ugens (follow the instructions in the ACEUGens folder)


// 2. Load server options and start server

(
// to evaluate, click here and hit Ctrl+Return
s = Server.local; // use local server
~path = PathName(thisProcess.nowExecutingPath); // current path
~loadfiles = {|targets| targets.do{ |target| (~path.pathOnly ++ ~path.fileNameWithoutExtension ++ "_" ++ target ++ ".scd").load } }; // function to load external files
~loadfiles.(\options); // set general server options and load presets
~setOptions.(s, \default); // set server options for specific audio interface as 2nd argument
s.reboot; // (re-)boot server
)


// 3. Load ACE and start the GUIs

(
// to evaluate, click here and hit Ctrl+Return
// this may take a while... (see post window for status)
~loadfiles.([\settings, \functions, \calculations, \synthdefs, \gui, \ctl]); // load functions and settings
g = (); // new dictionary for GUI windows
g.expertGUI = ~expertGUI.(); // start expert GUI
g.simpleGUI = ~simpleGUI.(); // start simple GUI
g.interGUI = ~interGUI.(); // start inter-stimulus ACE GUI
)



// optional: control via MIDI
~startmidi.(); // start midi
~stopmidi.(); // stop midi

