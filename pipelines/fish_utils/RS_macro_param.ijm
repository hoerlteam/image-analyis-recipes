// This macro script runs the RS (radial symmetry) FIJI plug-in on all the images in all the sub-directories of the defined dir
// After finding the best parameters using the RS plugin GUI interactive mode on one example image,
// You can run this macro script on the entire dataset.
// Just change the directory path, and the values of the parameters in the begining of the script

// You can run this script either in the ImageJ GUI or headless (also from cluster) using this command (linux):
// <FIJI/DIR/PATH>/ImageJ-linux64 --headless --run </PATH/TO/THIS/SCRIPT>/RS_macro.ijm &> </PATH/TO/WHERE/YOU/WANT/YOUR/LOGFILE>.log

// The detection result table will be saved to the same directory as each image it was calculated for.

// Retrieve command-line arguments
args = getArgument();
argArray = split(args, ",");

// Check if the number of arguments is correct
if (lengthOf(argArray) != 17) {
    print("Error: Invalid number of arguments provided!");
    exit(-1);
}

//////// Define RS parameters: //////////
dir = argArray[0];					// path to files to be processed
out_dir = argArray[1];					// path where to save the output csv
timeFile = argArray[2];					// location of file where runtimes will be saved
channel = argArray[16];					// which channel to do the detection in 

anisotropyCoefficient = argArray[3];
ransac = argArray[4];					// options are "RANSAC" (log value "SIMPLE") / "No RANSAC" / "MULTICONSENSU"
imMin = argArray[5]; 					// img min intensity
imMax = argArray[6]; 					// max intensity
sigmaDoG = argArray[7]; 
thresholdDoG = argArray[8];
supportRadius = argArray[9];
inlierRatio = argArray[10];				// meaning: min inlier ratio
maxError = argArray[11]; 				// meaning: max error range
intensityThreshold = argArray[12];  			// meaning: spot intensity threshold
bsMethod = argArray[13];				// Log file 0 / 1 / 2 / 3 / 4 options correspond to "No background subtraction" / "Mean" / "Median" / "RANSAC on Mean" / "RANSAC on Median"
bsMaxError = argArray[14];				// background subtraction param
bsInlierRatio = argArray[15];				// background subtraction param
useMultithread = "use_multithreading";			// Not from log file (only in advanced mode)! If you wish to use multithreading "use_multithreading", else "" (empty string)
numThreads = 40;					// multithread param
blockSizX = 128;                   		  	// multithread param
blockSizY = 128;					// multithread param
blockSizZ = 16;						// multithread param


print(anisotropyCoefficient,ransac,imMin,imMax,intensityThreshold);
///////////////////////////////////////////////////

ransac_sub = split(ransac, ' ');
ransac_sub = ransac_sub[0];

bsMethod_sub = split(bsMethod, ' ');
bsMethod_sub = bsMethod_sub[0];

setBatchMode(true);

///////////////////////////////////////////////////

walkFiles(dir);

// Find all files in subdirs:
function walkFiles(dir) {
	list = getFileList(dir);
	for (i=0; i<list.length; i++) {
		if (endsWith(list[i], "/"))
		   walkFiles(""+dir+list[i]);

		// If image file
		else  if (endsWith(list[i], "_ch" + channel + ".tif")) 
		   processImage(dir, list[i]);
	}
}

function processImage(dirPath, imName) {
	
	open("" + dirPath + imName);

	results_csv_path = "" + out_dir + "RadialSymmetry_results_" + imName + 
	"_aniso" + anisotropyCoefficient + 
	"ransac" + ransac_sub + 
	"imMin" + imMin +
	"imMax" + imMax +
	"sig" + sigmaDoG +
	"thr" + thresholdDoG + 
	"suppReg" + supportRadius + 
	"inRat" + inlierRatio +
	"maxErr" + maxError + 
	"intensThr" + intensityThreshold + 
	"bsMethod" + bsMethod_sub + 
	"bsMaxErr" + bsMaxError + 
	"bsInRat" + bsInlierRatio +
	".csv";


	RSparams =  "image=" + imName + 
	" mode=Advanced anisotropy=" + anisotropyCoefficient + " robust_fitting=[" + ransac + "] use_anisotropy" + 
	" image_min=" + imMin + " image_max=" + imMax + " sigma=" + sigmaDoG + " threshold=" + thresholdDoG + 
	" support=" + supportRadius + " min_inlier_ratio=" + inlierRatio + " max_error=" + maxError + " spot_intensity_threshold=" + intensityThreshold + 
	" background=[" + bsMethod + "] background_subtraction_max_error=" + bsMaxError + " background_subtraction_min_inlier_ratio=" + bsInlierRatio + 
	" results_file=[" + results_csv_path + "]" + 
	" " + useMultithread + " num_threads=" + numThreads + " block_size_x=" + blockSizX + " block_size_y=" + blockSizY + " block_size_z=" + blockSizZ;

	//print(RSparams);

	startTime = getTime();
    run("32-bit");
	run("RS-FISH", RSparams);
	exeTime = getTime() - startTime; //in miliseconds
	
	// Save exeTime to file:
	File.append(results_csv_path + "," + exeTime + "\n ", timeFile);

	// Close all windows:
	run("Close All");	
	while (nImages>0) { 
		selectImage(nImages); 
		close(); 
    } 
} 

