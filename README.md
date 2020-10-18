# CrowdAnalysis
Crowd analysis in OpenCV

//installation of openCV in the local repository
mvn install:install-file -Dfile="C:\opencv\build\java\opencv-330.jar" -DgroupId=opencv -DartifactId=opencv -Dversion=330 -Dpackaging=jar

//Setting folder from which native library is loaded (under VM Options)
//System.setProperty(...) doesn't work because JVM parameters cannot change after it's been started, only their copies.
-Djava.library.path=C:\opencv\build\java\x64
