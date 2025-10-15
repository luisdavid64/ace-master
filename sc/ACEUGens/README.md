# ACE UGens

## Installation

Releases are available from the [release page](https://git.iem.at/weger/ace/-/releases).

Unzip the release and move it to your SuperCollider extensions folder. You can find it by evaluating

```supercollider
Platform.userExtensionDir
```

in SuperCollider. To evaluate code in SuperCollder, put your cursor on the line of code and press `Cmd+Enter`
(macOS) or `Ctrl+Enter`.  Alternatively, you may install the extensions system-wide by copying to

```supercollider
Platform.systemExtensionDir
```

The folder might not exist, so you may need to create it yourself. You can do this in your operating system's file
explorer or from within SuperCollider by evaluating:

```supercollider
File.mkdir(Platform.userExtensionDir)
```

On some operating systems, these directories may be hard to find because they're in hidden folders.  You can open
the user app support directory (where the Extensions folder is) with the menu item
"File->Open user support directory". On macOS, you can open a finder window and press `Cmd+Shift+G` and enter the
name of the directory.

## Compile from source

This is how you build the plugins in this directory.

### Step 1: Obtain header files

Before you can compile any plugin, you will need a copy of the SuperCollider *source code* (NOT the app itself). 
Source code tarballs can be downloaded from the [SuperCollider release page](https://github.com/supercollider/supercollider/releases). If you are on Linux, it's okay (and preferable) to use the Linux source tarball.

You will **not** need to recompile SuperCollider itself in order to get a plugin working. You only need the source code to get the C++ headers.

The source code version should roughly match your SuperCollider app version. For example, headers from any 3.9.x patch release will produce plugins compatible with any 3.9.x version, but not 3.8. This is due to occasional breaking changes in the plugin "API" (technically the ABI), which will occur only in 3.x releases. These breaking changes will not require modification to your plugin's source code, but compiled plugin binaries will need to be recompiled. If the server tries to load an incompatible plugin, it will give the "API version mismatch" error message.

### Step 2: Create build directory, set `SC_PATH` and flags

CMake dumps a lot of files into your working directory, so you should always start by creating the `build/` directory:

```shell
ace/ACEUGens/$ mkdir build
ace/ACEUGens/$ cd build
```

Next, we run CMake and tell it where the SuperCollider headers are to be found (don't forget the `..`!):

```shell
ace/ACEUGens/build/$ cmake -DSC_PATH=/path/to/supercollider/ ..
```

Under WINDOWS use the absolute path, and add VS version:

```shell
ace/ACEUGens/build/$ cmake -G "Visual Studio 15 2017 Win64" -DSC_PATH=C:\path\to\supercollider\ ..
```

Here, `/path/to/supercollider/` is the path to the source code. On Linux, it may be located in `/usr/include/SuperCollider/`. Once again, this is the *source code*, not the app itself.

To make sure you have the right path, check to ensure that it contains a file at `include/plugin_interface/SC_PlugIn.h`. If you get a warning that `SC_PlugIn.h` could not be found, then `SC_PATH` is not set correctly.



If you don't plan on using a debugger, it's advisable to build in release mode so that the compiler optimizes. For LINUX, type:

```shell
ace/ACEUGens/build/$ cmake -DCMAKE_BUILD_TYPE=RELEASE ..
```

Switch back to debug mode with `-DCMAKE_BUILD_TYPE=DEBUG`.


Again, all these flags are persistent, and you only need to run them once. If something is messed up, you can trash the `build/` directory and start again.

### Step 3: Build it!

After that, make sure you're in the build directory again.  

For LINUX, just call `make`:

```shell
ace/ACEUGens/build/$ make
```

For WINDOWS, a Visual Studio solution has been created. Run it in Visual Studio, select "Release" and build it.

```shell
ace/ACEUGens/build/$ run ACEUGens.sln
```

This will produce a "shared library" file ending in `.scx`. On Linux, the extension is `.so`.


Now copy, move, or symbolic link the folder into your Extensions folder. Follow the installation instructions above.
