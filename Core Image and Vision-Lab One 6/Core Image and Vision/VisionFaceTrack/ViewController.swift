/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Contains the main app implementation using Vision.
*/

import UIKit
import AVKit
import Vision

class ViewController: UIViewController {
    
    // Main view for showing camera content.
    @IBOutlet weak var previewView: UIView?
    @IBOutlet weak var gestureLabel: UILabel!
    // AVCapture variables to hold sequence data
    var session: AVCaptureSession?
    var previewLayer: AVCaptureVideoPreviewLayer?
    
    var playerScore = 0
    var cpuScore = 0
        
    var videoDataOutput: AVCaptureVideoDataOutput?
    var videoDataOutputQueue: DispatchQueue?
    
    var captureDevice: AVCaptureDevice?
    var captureDeviceResolution: CGSize = CGSize()
    
    // Layer UI for drawing Vision results
    var rootLayer: CALayer?
    var detectionOverlayLayer: CALayer?
    var detectedFaceRectangleShapeLayer: CAShapeLayer?
    var detectedFaceLandmarksShapeLayer: CAShapeLayer?
    var detectedHandShapeLayer: CAShapeLayer?
    
    // Vision requests
    private var detectionRequests: [VNRequest]?
    private var trackingRequests: [VNTrackObjectRequest]?
    
    lazy var sequenceRequestHandler = VNSequenceRequestHandler()
    
    // MARK: UIViewController overrides
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Setup capture session and vision request
        self.session = self.setupAVCaptureSession()
        self.prepareVisionRequest()
        
        // Start the capture session
        self.session?.startRunning()
        
        // Ensure gestureLabel is brought to the front
        if let gestureLabel = self.gestureLabel {
            self.view.bringSubviewToFront(gestureLabel)
        }
    }
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    // Ensure that the interface stays locked in Portrait.
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        return .portrait
    }
    
    // Ensure that the interface stays locked in Portrait.
    override var preferredInterfaceOrientationForPresentation: UIInterfaceOrientation {
        return .portrait
    }
    
    
    
    
    // MARK: Performing Vision Requests
    
    /// - Tag: WriteCompletionHandler
    fileprivate func prepareVisionRequest() {
        self.trackingRequests = []
        
        // Create hand pose detection request
        let handPoseRequest = VNDetectHumanHandPoseRequest(completionHandler: self.handPoseDetectionCompletionHandler)
        self.detectionRequests = [handPoseRequest]  // Now it works because detectionRequests is [VNRequest] type
        
        self.sequenceRequestHandler = VNSequenceRequestHandler()
        
        // Setup drawing layers for showing the output of hand pose detection
        self.setupVisionDrawingLayers()
    }
    
    /// Handles the results of a hand pose detection request, identifies gestures, and updates the UI.
    ///
    /// - Parameters:
    ///   - request: The `VNRequest` containing the hand pose detection results.
    ///   - error: An optional `Error` if the request failed.
    func handPoseDetectionCompletionHandler(request: VNRequest, error: Error?) {
        guard let results = request.results as? [VNHumanHandPoseObservation] else { return }

        let path = CGMutablePath()
        var detectedGesture: String = "Unknown"

        for observation in results {
            detectedGesture = recognizeHandGesture(observation)
            
            // Draw rectangle around detected key points for visualization
            if let thumbTip = try? observation.recognizedPoints(.thumb)[.thumbTip],
               let indexTip = try? observation.recognizedPoints(.indexFinger)[.indexTip] {
                let thumbPoint = CGPoint(x: thumbTip.location.x * self.view.bounds.width,
                                         y: (1 - thumbTip.location.y) * self.view.bounds.height)
                let indexPoint = CGPoint(x: indexTip.location.x * self.view.bounds.width,
                                         y: (1 - indexTip.location.y) * self.view.bounds.height)
                path.addRect(CGRect(x: min(thumbPoint.x, indexPoint.x),
                                    y: min(thumbPoint.y, indexPoint.y),
                                    width: abs(thumbPoint.x - indexPoint.x),
                                    height: abs(thumbPoint.y - indexPoint.y)))
            }
        }

        DispatchQueue.main.async {
            CATransaction.begin()
            CATransaction.setAnimationDuration(0.2)
            self.detectedHandShapeLayer?.path = path
            
            // Display UI feedback for the detected gesture
            switch detectedGesture {
            case "Rock":
                self.displayRockUI()
            case "Scissors":
                self.displayScissorsUI()
            case "Paper":
                self.displayPaperUI()
            default:
                break
            }
            
            // Trigger a match against the CPU with the exact detected gesture
            if detectedGesture != "Unknown" {
                self.playAgainstCPU(playerGesture: detectedGesture)
            }

            CATransaction.commit()
        }
    }       
    // define behavior for when we detect a face
    func faceDetectionCompletionHandler(request:VNRequest, error: Error?){
        // any errors? If yes, show and try to keep going
        if error != nil {
            print("FaceDetection error: \(String(describing: error)).")
        }
        
        // see if we can get any face features, this will fail if no faces detected
        // try to save the face observations to a results vector
        guard let faceDetectionRequest = request as? VNDetectFaceRectanglesRequest,
            let results = faceDetectionRequest.results as? [VNFaceObservation] else {
                return
        }
        
        if !results.isEmpty{
            print("Initial Face found... setting up tracking.")
            
            
        }
        
        // if we got here, then a face was detected and we have its features saved
        // The above face detection was the most computational part of what we did
        // the remaining tracking only needs the results vector of face features
        // so we can process it in the main queue (because we will us it to update UI)
        DispatchQueue.main.async {
            // Add the face features to the tracking list
            for observation in results {
                let faceTrackingRequest = VNTrackObjectRequest(detectedObjectObservation: observation)
                // the array starts empty, but this will constantly add to it
                // since on the main queue, there are no race conditions
                // everything is from a single thread
                // once we add this, it kicks off tracking in another function
                self.trackingRequests?.append(faceTrackingRequest)
                
                // NOTE: if the initial face detection is actually not a face,
                // then the app will continually mess up trying to perform tracking
            }
        }
        
    }
    
    
    // MARK: AVCaptureVideoDataOutputSampleBufferDelegate
    /// - Tag: PerformRequests
    // Handle delegate method callback on receiving a sample buffer.
    // This is where we get the pixel buffer from the camera and need to
    // generate the vision requests
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) 
    {
        
        var requestHandlerOptions: [VNImageOption: AnyObject] = [:]
        
        // see if camera has any instrinsic transforms on it
        // if it does, add these to the options for requests
        let cameraIntrinsicData = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil)
        if cameraIntrinsicData != nil {
            requestHandlerOptions[VNImageOption.cameraIntrinsics] = cameraIntrinsicData
        }
        
        // check to see if we can get the pixels for processing, else return
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to obtain a CVPixelBuffer for the current output frame.")
            return
        }
        
        // get portrait orientation for UI
        let exifOrientation = self.exifOrientationForCurrentDeviceOrientation()
        
        guard let requests = self.trackingRequests else {
            print("Tracking request array not setup, aborting.")
            return
        }

        
        // check to see if the tracking request is empty (no face currently detected)
        // if it is empty,
        if requests.isEmpty{
            // No tracking object detected, so perform initial detection
            // the initial detection takes some time to perform
            // so we special case it here
            
            self.performInitialDetection(pixelBuffer: pixelBuffer,
                                         exifOrientation: exifOrientation,
                                         requestHandlerOptions: requestHandlerOptions)
            
            return  // just perform the initial request
        }
        
        // if tracking was not empty, it means we have detected a face very recently
        // so no we can process the sequence of tracking face features
        
        self.performTracking(requests: requests,
                             pixelBuffer: pixelBuffer,
                             exifOrientation: exifOrientation)
        
        
        // if there are no valid observations, then this will be empty
        // the function above will empty out all the elements
        // in our tracking if nothing is high confidence in the output
        if let newTrackingRequests = self.trackingRequests {
            
            if newTrackingRequests.isEmpty {
                // Nothing was high enough confidence to track, just abort.
                print("Face object lost, resetting detection...")
                return
            }

            self.performLandmarkDetection(newTrackingRequests: newTrackingRequests,
                                          pixelBuffer: pixelBuffer,
                                          exifOrientation: exifOrientation,
                                          requestHandlerOptions: requestHandlerOptions)
  
        }
    
        
    }
    
    // functionality to run the image detection on pixel buffer
    // This is an involved computation, so beware of running too often
    
    func performInitialHandPoseDetection(pixelBuffer: CVPixelBuffer, exifOrientation: CGImagePropertyOrientation, requestHandlerOptions: [VNImageOption: AnyObject]) {
        // Initialize the VNImageRequestHandler with the provided pixel buffer and options
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                        orientation: exifOrientation,
                                                        options: requestHandlerOptions)
        do {
            // Execute the detection requests if they are available
            if let detectRequests = self.detectionRequests {
                try imageRequestHandler.perform(detectRequests)
            }
        } catch let error as NSError {
            // Log an error if the detection fails
            NSLog("Failed to perform HandPoseRequest: %@", error)
        }
    }
    
    /// Recognizes a hand gesture (Rock, Paper, or Scissors) based on finger tip positions.
    ///
    /// - Parameter observation: A `VNHumanHandPoseObservation` containing recognized hand points.
    /// - Returns: A `String` representing the recognized gesture: "Rock", "Paper", or "Scissors".

    func recognizeHandGesture(_ observation: VNHumanHandPoseObservation) -> String {
        // Try to retrieve the recognized points for thumb, index, and middle fingers.
        // If any of these points is not available, return "Unknown" as the gesture.
        guard let thumbTip = try? observation.recognizedPoints(.thumb)[.thumbTip],
              let indexTip = try? observation.recognizedPoints(.indexFinger)[.indexTip],
              let middleTip = try? observation.recognizedPoints(.middleFinger)[.middleTip] else {
            return "Unknown"
        }
        
        // Calculate the distance between the index and middle finger tips.
        // This helps to determine if the gesture might be "Rock".
        let indexMiddleDistance = hypot(indexTip.location.x - middleTip.location.x,
                                        indexTip.location.y - middleTip.location.y)
        
        // Determine the gesture based on finger positions.
        if indexMiddleDistance < 0.1 {
            // If the index and middle fingers are close to each other, assume "Rock".
            return "Rock"
        } else if thumbTip.location.x < indexTip.location.x && thumbTip.location.x < middleTip.location.x {
            // If the thumb is positioned to the left of both the index and middle fingers,
            // assume "Paper".
            return "Paper"
        } else {
            // Otherwise, assume "Scissors" if none of the above conditions are met.
            return "Scissors"
        }
    }
    // Function to display Rock gesture UI
    func displayRockUI() {
        updateOverlayColor(to: .blue)
        showUIFeedback(labelText: "✊ Rock Detected!", color: .blue, positionY: 100)
    }

    // Function to display Scissors gesture UI
    func displayScissorsUI() {
        updateOverlayColor(to: .red)
        showUIFeedback(labelText: "✌️ Scissors Detected!", color: .red, positionY: 200)
    }

    // Function to display Paper gesture UI
    func displayPaperUI() {
        updateOverlayColor(to: .green)
        showUIFeedback(labelText: "🖐 Paper Detected!", color: .green, positionY: 300)
    }
    // Function to update overlay color
    private func updateOverlayColor(to color: UIColor) {
        detectionOverlayLayer?.sublayers?.forEach { layer in
            if let shapeLayer = layer as? CAShapeLayer {
                shapeLayer.strokeColor = color.cgColor
            }
        }
    }

    // Function to display animated feedback text on the UI
    private func showUIFeedback(labelText: String, color: UIColor, positionY: CGFloat) {
        let label = UILabel()
        label.text = labelText
        label.font = UIFont.systemFont(ofSize: 24)
        label.textColor = color
        label.alpha = 0 // Start label as invisible
        label.frame = CGRect(x: 50, y: positionY, width: 250, height: 50)
        self.view.addSubview(label)
        
        // Animate the label's appearance and disappearance
        UIView.animate(withDuration: 0.5, animations: {
            label.alpha = 1 // Fade in
        }) { _ in
            UIView.animate(withDuration: 1.0, delay: 1.0, options: [], animations: {
                label.alpha = 0 // Fade out
            }) { _ in
                label.removeFromSuperview() // Remove the label after fading out
            }
        }
    }
    func updateUIForGesture(_ gesture: String) {
        DispatchQueue.main.async {
            // Ensure gestureLabel is visible and brought to the front
            if let gestureLabel = self.gestureLabel {
                self.view.bringSubviewToFront(gestureLabel)
                
                // Set gesture label text based on the gesture type
                switch gesture {
                case "Scissors":
                    gestureLabel.text = "✌️ Scissors Detected!"
                    self.detectedHandShapeLayer?.strokeColor = UIColor.red.cgColor
                case "Rock":
                    gestureLabel.text = "✊ Rock Detected!"
                    self.detectedHandShapeLayer?.strokeColor = UIColor.blue.cgColor
                case "Paper":
                    gestureLabel.text = "🖐 Paper Detected!"
                    self.detectedHandShapeLayer?.strokeColor = UIColor.green.cgColor
                default:
                    gestureLabel.text = ""
                    self.detectedHandShapeLayer?.strokeColor = UIColor.gray.cgColor
                }
                
                // Optional: Add animation to make the label more noticeable
                UIView.animate(withDuration: 0.2, animations: {
                    gestureLabel.alpha = 1.0
                }) { _ in
                    UIView.animate(withDuration: 0.5, delay: 1.0, options: [], animations: {
                        gestureLabel.alpha = 0.0
                    }, completion: nil)
                }
                
                print("Detected Gesture: \(gesture)")
            }
        }
    }
    
    // Play a round against the CPU, updating scores and displaying results
    func playAgainstCPU(playerGesture: String) {
        let cpuGesture = generateCPUGesture()  // Generate CPU's gesture
        let result = determineWinner(playerGesture: playerGesture, cpuGesture: cpuGesture)  // Determine winner
        updateScore(result: result)  // Update scores
        displayResult(playerGesture: playerGesture, cpuGesture: cpuGesture, result: result)  // Show result
    }

    // Randomly generates a gesture for the CPU
    func generateCPUGesture() -> String {
        let gestures = ["Rock", "Paper", "Scissors"]
        return gestures[Int.random(in: 0...2)]
    }

    // Determines the winner between player and CPU gestures
    func determineWinner(playerGesture: String, cpuGesture: String) -> String {
        if playerGesture == cpuGesture {
            return "Draw"
        }
        
        switch (playerGesture, cpuGesture) {
        case ("Rock", "Scissors"), ("Scissors", "Paper"), ("Paper", "Rock"):
            return "Player Wins"
        default:
            return "CPU Wins"
        }
    }

    // Updates the score based on the result of the round
    func updateScore(result: String) {
        switch result {
        case "Player Wins":
            playerScore += 1
        case "CPU Wins":
            cpuScore += 1
        default:
            break
        }
        print("Current Score - Player: \(playerScore), CPU: \(cpuScore)")
    }

    // Displays the result of the match
    func displayResult(playerGesture: String, cpuGesture: String, result: String) {
        let message = """
        Player Gesture: \(playerGesture)
        CPU Gesture: \(cpuGesture)
        
        Result: \(result)
        """
        let alertController = UIAlertController(title: "Round Result", message: message, preferredStyle: .alert)
        alertController.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
        self.present(alertController, animated: true, completion: nil)
    }
    func performInitialDetection(pixelBuffer:CVPixelBuffer, exifOrientation:CGImagePropertyOrientation, requestHandlerOptions:[VNImageOption: AnyObject]) {
        // create request
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                        orientation: exifOrientation,
                                                        options: requestHandlerOptions)
        
        do {
            if let detectRequests = self.detectionRequests{
                // try to detect face and add it to tracking buffer
                try imageRequestHandler.perform(detectRequests)
            }
        } catch let error as NSError {
            NSLog("Failed to perform FaceRectangleRequest: %@", error)
        }
    }
    
    
    // this function performs all the tracking of the face sequence
    func performTracking(requests:[VNTrackObjectRequest],
                         pixelBuffer:CVPixelBuffer, exifOrientation:CGImagePropertyOrientation)
    {
        do {
            // perform tracking on the pixel buffer, which is
            // less computational than fully detecting a face
            // if a face was not correct initially, this tracking
            //   will also be not great... but it is fast!
            try self.sequenceRequestHandler.perform(requests,
                                                    on: pixelBuffer,
                                                    orientation: exifOrientation)
        } catch let error as NSError {
            NSLog("Failed to perform SequenceRequest: %@", error)
        }
        
        // if there are any tracking results, let's process them here
        
        // Setup the next round of tracking.
        var newTrackingRequests = [VNTrackObjectRequest]()
        for trackingRequest in requests {
            
            // any valid results in the request?
            // if so, grab the first request
            if let results = trackingRequest.results,
               let observation = results[0] as? VNDetectedObjectObservation {
                
                
                // is this tracking request of high confidence?
                // If it is, then we should add it to processing buffer
                // the threshold is arbitrary. You can adjust to you liking
                if !trackingRequest.isLastFrame {
                    if observation.confidence > 0.3 {
                        trackingRequest.inputObservation = observation
                    }
                    else {

                        // once below thresh, make it last frame
                        // this will stop the processing of tracker
                        trackingRequest.isLastFrame = true
                    }
                    // add to running tally of high confidence observations
                    newTrackingRequests.append(trackingRequest)
                }
                
            }
            
        }
        self.trackingRequests = newTrackingRequests
        
        
    }
    
    func performLandmarkDetection(newTrackingRequests:[VNTrackObjectRequest], pixelBuffer:CVPixelBuffer, exifOrientation:CGImagePropertyOrientation, requestHandlerOptions:[VNImageOption: AnyObject]) {
        // Perform face landmark tracking on detected faces.
        // setup an empty arry for now
        var faceLandmarkRequests = [VNDetectFaceLandmarksRequest]()
        
        // Perform landmark detection on tracked faces.
        for trackingRequest in newTrackingRequests {
            
            // create a request for facial landmarks
            let faceLandmarksRequest = VNDetectFaceLandmarksRequest(completionHandler: self.landmarksCompletionHandler)
            
            // get tracking result and observation for result
            if let trackingResults = trackingRequest.results,
               let observation = trackingResults[0] as? VNDetectedObjectObservation{
                
                // save the observation info
                let faceObservation = VNFaceObservation(boundingBox: observation.boundingBox)
                
                // set information for face
                faceLandmarksRequest.inputFaceObservations = [faceObservation]
                
                // Continue to track detected facial landmarks.
                faceLandmarkRequests.append(faceLandmarksRequest)
                
                // setup for performing landmark detection
                let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                                orientation: exifOrientation,
                                                                options: requestHandlerOptions)
                
                do {
                    // try to find landmarks in face, then display in completion handler
                    try imageRequestHandler.perform(faceLandmarkRequests)
                    
                    // completion handler will now take over and finish the job!
                } catch let error as NSError {
                    NSLog("Failed to perform FaceLandmarkRequest: %@", error)
                }
            }
        }
    }
    
    
    // Interpret the output of our facial landmark detector
    // this code is called upon succesful completion of landmark detection
    func landmarksCompletionHandler(request:VNRequest, error:Error?){
        
        if error != nil {
            print("FaceLandmarks error: \(String(describing: error)).")
        }
        
        // any landmarks found that we can display? If not, return
        guard let landmarksRequest = request as? VNDetectFaceLandmarksRequest,
              let results = landmarksRequest.results as? [VNFaceObservation] else {
            return
        }
        
        // Perform all UI updates (drawing) on the main queue, not the background queue on which this handler is being called.
        DispatchQueue.main.async {
            // draw the landmarks using core animation layers
            self.drawFaceObservations(results)
        }
    }
    
    
}


// MARK: Helper Methods
extension UIViewController{
    
    // Helper Methods for Error Presentation
    
    fileprivate func presentErrorAlert(withTitle title: String = "Unexpected Failure", message: String) {
        let alertController = UIAlertController(title: title, message: message, preferredStyle: .alert)
        self.present(alertController, animated: true)
    }
    
    fileprivate func presentError(_ error: NSError) {
        self.presentErrorAlert(withTitle: "Failed with error \(error.code)", message: error.localizedDescription)
    }
    
    // Helper Methods for Handling Device Orientation & EXIF
    
    fileprivate func radiansForDegrees(_ degrees: CGFloat) -> CGFloat {
        return CGFloat(Double(degrees) * Double.pi / 180.0)
    }
    
    func exifOrientationForDeviceOrientation(_ deviceOrientation: UIDeviceOrientation) -> CGImagePropertyOrientation {
        
        switch deviceOrientation {
        case .portraitUpsideDown:
            return .rightMirrored
            
        case .landscapeLeft:
            return .downMirrored
            
        case .landscapeRight:
            return .upMirrored
            
        default:
            return .leftMirrored
        }
    }
    
    func exifOrientationForCurrentDeviceOrientation() -> CGImagePropertyOrientation {
        return exifOrientationForDeviceOrientation(UIDevice.current.orientation)
    }
}


// MARK: Extension for AVCapture Setup
extension ViewController:AVCaptureVideoDataOutputSampleBufferDelegate{
    
    
    /// - Tag: CreateCaptureSession
    fileprivate func setupAVCaptureSession() -> AVCaptureSession? {
        let captureSession = AVCaptureSession()
        do {
            let inputDevice = try self.configureFrontCamera(for: captureSession)
            self.configureVideoDataOutput(for: inputDevice.device, resolution: inputDevice.resolution, captureSession: captureSession)
            self.designatePreviewLayer(for: captureSession)
            return captureSession
        } catch let executionError as NSError {
            self.presentError(executionError)
        } catch {
            self.presentErrorAlert(message: "An unexpected failure has occured")
        }
        
        self.teardownAVCapture()
        
        return nil
    }
    
    /// - Tag: ConfigureDeviceResolution
    fileprivate func highestResolution420Format(for device: AVCaptureDevice) -> (format: AVCaptureDevice.Format, resolution: CGSize)? {
        var highestResolutionFormat: AVCaptureDevice.Format? = nil
        var highestResolutionDimensions = CMVideoDimensions(width: 0, height: 0)
        
        for format in device.formats {
            let deviceFormat = format as AVCaptureDevice.Format
            
            let deviceFormatDescription = deviceFormat.formatDescription
            if CMFormatDescriptionGetMediaSubType(deviceFormatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
                let candidateDimensions = CMVideoFormatDescriptionGetDimensions(deviceFormatDescription)
                if (highestResolutionFormat == nil) || (candidateDimensions.width > highestResolutionDimensions.width) {
                    highestResolutionFormat = deviceFormat
                    highestResolutionDimensions = candidateDimensions
                }
            }
        }
        
        if highestResolutionFormat != nil {
            let resolution = CGSize(width: CGFloat(highestResolutionDimensions.width), height: CGFloat(highestResolutionDimensions.height))
            return (highestResolutionFormat!, resolution)
        }
        
        return nil
    }
    
    fileprivate func configureFrontCamera(for captureSession: AVCaptureSession) throws -> (device: AVCaptureDevice, resolution: CGSize) {
        let deviceDiscoverySession = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .front)
        
        if let device = deviceDiscoverySession.devices.first {
            if let deviceInput = try? AVCaptureDeviceInput(device: device) {
                if captureSession.canAddInput(deviceInput) {
                    captureSession.addInput(deviceInput)
                }
                
                if let highestResolution = self.highestResolution420Format(for: device) {
                    try device.lockForConfiguration()
                    device.activeFormat = highestResolution.format
                    device.unlockForConfiguration()
                    
                    return (device, highestResolution.resolution)
                }
            }
        }
        
        throw NSError(domain: "ViewController", code: 1, userInfo: nil)
    }
    
    /// - Tag: CreateSerialDispatchQueue
    fileprivate func configureVideoDataOutput(for inputDevice: AVCaptureDevice, resolution: CGSize, captureSession: AVCaptureSession) {
        
        let videoDataOutput = AVCaptureVideoDataOutput()
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        
        // Create a serial dispatch queue used for the sample buffer delegate as well as when a still image is captured.
        // A serial dispatch queue must be used to guarantee that video frames will be delivered in order.
        let videoDataOutputQueue = DispatchQueue(label: "com.example.apple-samplecode.VisionFaceTrack")
        videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
        }
        
        videoDataOutput.connection(with: .video)?.isEnabled = true
        
        if let captureConnection = videoDataOutput.connection(with: AVMediaType.video) {
            if captureConnection.isCameraIntrinsicMatrixDeliverySupported {
                captureConnection.isCameraIntrinsicMatrixDeliveryEnabled = true
            }
        }
        
        self.videoDataOutput = videoDataOutput
        self.videoDataOutputQueue = videoDataOutputQueue
        
        self.captureDevice = inputDevice
        self.captureDeviceResolution = resolution
    }
    
    /// - Tag: DesignatePreviewLayer
    fileprivate func designatePreviewLayer(for captureSession: AVCaptureSession) {
        let videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        self.previewLayer = videoPreviewLayer
        
        videoPreviewLayer.name = "CameraPreview"
        videoPreviewLayer.backgroundColor = UIColor.black.cgColor
        videoPreviewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        
        if let previewRootLayer = self.previewView?.layer {
            self.rootLayer = previewRootLayer
            
            previewRootLayer.masksToBounds = true
            videoPreviewLayer.frame = previewRootLayer.bounds
            previewRootLayer.addSublayer(videoPreviewLayer)
        }
    }
    
    // Removes infrastructure for AVCapture as part of cleanup.
    fileprivate func teardownAVCapture() {
        self.videoDataOutput = nil
        self.videoDataOutputQueue = nil
        
        if let previewLayer = self.previewLayer {
            previewLayer.removeFromSuperlayer()
            self.previewLayer = nil
        }
    }
}


// MARK: Extension Drawing Vision Observations
extension ViewController {
    
    
    fileprivate func setupVisionDrawingLayers() {
        // Initialize detection overlay
        let overlayLayer = CALayer()
        overlayLayer.name = "DetectionOverlay"
        overlayLayer.bounds = self.view.bounds
        overlayLayer.position = CGPoint(x: self.view.bounds.midX, y: self.view.bounds.midY)
        self.view.layer.addSublayer(overlayLayer)
        self.detectionOverlayLayer = overlayLayer
        
        // Initialize shape layer for hand detection
        let shapeLayer = CAShapeLayer()
        shapeLayer.bounds = overlayLayer.bounds
        shapeLayer.position = CGPoint(x: overlayLayer.bounds.midX, y: overlayLayer.bounds.midY)
        shapeLayer.strokeColor = UIColor.red.cgColor  // Default color
        shapeLayer.lineWidth = 3.0
        shapeLayer.fillColor = UIColor.clear.cgColor
        overlayLayer.addSublayer(shapeLayer)
        self.detectedHandShapeLayer = shapeLayer
    }
    
    fileprivate func updateLayerGeometry() {
        guard let overlayLayer = self.detectionOverlayLayer,
            let rootLayer = self.rootLayer,
            let previewLayer = self.previewLayer
            else {
            return
        }
        
        CATransaction.setValue(NSNumber(value: true), forKey: kCATransactionDisableActions)
        
        let videoPreviewRect = previewLayer.layerRectConverted(fromMetadataOutputRect: CGRect(x: 0, y: 0, width: 1, height: 1))
        
        var rotation: CGFloat
        var scaleX: CGFloat
        var scaleY: CGFloat
        
        // Rotate the layer into screen orientation.
        switch UIDevice.current.orientation {
        case .portraitUpsideDown:
            rotation = 180
            scaleX = videoPreviewRect.width / captureDeviceResolution.width
            scaleY = videoPreviewRect.height / captureDeviceResolution.height
            
        case .landscapeLeft:
            rotation = 90
            scaleX = videoPreviewRect.height / captureDeviceResolution.width
            scaleY = scaleX
            
        case .landscapeRight:
            rotation = -90
            scaleX = videoPreviewRect.height / captureDeviceResolution.width
            scaleY = scaleX
            
        default:
            rotation = 0
            scaleX = videoPreviewRect.width / captureDeviceResolution.width
            scaleY = videoPreviewRect.height / captureDeviceResolution.height
        }
        
        // Scale and mirror the image to ensure upright presentation.
        let affineTransform = CGAffineTransform(rotationAngle: radiansForDegrees(rotation))
            .scaledBy(x: scaleX, y: -scaleY)
        overlayLayer.setAffineTransform(affineTransform)
        
        // Cover entire screen UI.
        let rootLayerBounds = rootLayer.bounds
        overlayLayer.position = CGPoint(x: rootLayerBounds.midX, y: rootLayerBounds.midY)
    }
    
    fileprivate func addPoints(in landmarkRegion: VNFaceLandmarkRegion2D, to path: CGMutablePath, applying affineTransform: CGAffineTransform, closingWhenComplete closePath: Bool) {
        let pointCount = landmarkRegion.pointCount
        if pointCount > 1 {
            let points: [CGPoint] = landmarkRegion.normalizedPoints
            path.move(to: points[0], transform: affineTransform)
            path.addLines(between: points, transform: affineTransform)
            if closePath {
                path.addLine(to: points[0], transform: affineTransform)
                path.closeSubpath()
            }
        }
    }
    
    fileprivate func addIndicators(to faceRectanglePath: CGMutablePath, faceLandmarksPath: CGMutablePath, for faceObservation: VNFaceObservation) {
        let displaySize = self.captureDeviceResolution
        
        let faceBounds = VNImageRectForNormalizedRect(faceObservation.boundingBox, Int(displaySize.width), Int(displaySize.height))
        faceRectanglePath.addRect(faceBounds)
        
        if let landmarks = faceObservation.landmarks {
            // Landmarks are relative to -- and normalized within --- face bounds
            let affineTransform = CGAffineTransform(translationX: faceBounds.origin.x, y: faceBounds.origin.y)
                .scaledBy(x: faceBounds.size.width, y: faceBounds.size.height)
            
            // Treat eyebrows and lines as open-ended regions when drawing paths.
            let openLandmarkRegions: [VNFaceLandmarkRegion2D?] = [
                landmarks.leftEyebrow,
                landmarks.rightEyebrow,
                landmarks.faceContour,
                landmarks.noseCrest,
                landmarks.medianLine
            ]
            for openLandmarkRegion in openLandmarkRegions where openLandmarkRegion != nil {
                self.addPoints(in: openLandmarkRegion!, to: faceLandmarksPath, applying: affineTransform, closingWhenComplete: false)
            }
            
            // Draw eyes, lips, and nose as closed regions.
            let closedLandmarkRegions: [VNFaceLandmarkRegion2D?] = [
                landmarks.leftEye,
                landmarks.rightEye,
                landmarks.outerLips,
                landmarks.innerLips,
                landmarks.nose
            ]
            for closedLandmarkRegion in closedLandmarkRegions where closedLandmarkRegion != nil {
                self.addPoints(in: closedLandmarkRegion!, to: faceLandmarksPath, applying: affineTransform, closingWhenComplete: true)
            }
        }
    }
    
    /// - Tag: DrawPaths
    fileprivate func drawFaceObservations(_ faceObservations: [VNFaceObservation]) {
        guard let faceRectangleShapeLayer = self.detectedFaceRectangleShapeLayer,
            let faceLandmarksShapeLayer = self.detectedFaceLandmarksShapeLayer
            else {
            return
        }
        
        CATransaction.begin()
        
        CATransaction.setValue(NSNumber(value: true), forKey: kCATransactionDisableActions)
        
        let faceRectanglePath = CGMutablePath()
        let faceLandmarksPath = CGMutablePath()
        
        for faceObservation in faceObservations {
            self.addIndicators(to: faceRectanglePath,
                               faceLandmarksPath: faceLandmarksPath,
                               for: faceObservation)
        }
        
        faceRectangleShapeLayer.path = faceRectanglePath
        faceLandmarksShapeLayer.path = faceLandmarksPath
        
        self.updateLayerGeometry()
        
        CATransaction.commit()
    }
}





    
