import SwiftUI
import HealthKit
import Speech
import AVFoundation  // Import for Text-to-Speech

class SpeechRecognizer: ObservableObject {
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))!
    private let audioEngine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    
    @Published var recognizedText = ""
    @Published var isRecording = false
    
    init() {
        // Request authorization to use Speech Recognition
        SFSpeechRecognizer.requestAuthorization { authStatus in
            if authStatus != .authorized {
                print("Speech recognition not authorized")
            }
        }
    }
    
    func startRecording() {
        guard !audioEngine.isRunning else {
            stopRecording()
            return
        }
        
        recognizedText = "" // Clear previous recognition results
        isRecording = true
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { (buffer, _) in
            self.request?.append(buffer)
        }
        
        do {
            try audioEngine.start()
            request = SFSpeechAudioBufferRecognitionRequest()
            task = speechRecognizer.recognitionTask(with: request!, resultHandler: { result, error in
                if let result = result {
                    self.recognizedText = result.bestTranscription.formattedString
                }
                if let error = error {
                    print("Error during recognition: \(error.localizedDescription)")
                }
            })
        } catch {
            print("Audio engine couldn't start: \(error.localizedDescription)")
        }
    }
    
    func stopRecording() {
        audioEngine.stop()
        request?.endAudio()
        isRecording = false
    }
}

struct SpeechToTextView: View {
    @StateObject private var speechRecognizer = SpeechRecognizer()
    @Binding var userContext: String  // Accept userContext as a binding
    
    var body: some View {
        VStack {
            Text(speechRecognizer.recognizedText)
                .padding()
                .frame(height: 100)
                .border(Color.gray)
            
            Button(action: {
                if speechRecognizer.isRecording {
                    speechRecognizer.stopRecording()
                } else {
                    speechRecognizer.startRecording()
                }
            }) {
                Image(systemName: speechRecognizer.isRecording ? "mic.circle.fill" : "mic.circle")
                                    .resizable()
                                    .frame(width: 50, height: 50)
                                    .foregroundColor(speechRecognizer.isRecording ? .red : .blue)
            }
        }
        .padding()
        .onChange(of: speechRecognizer.recognizedText) { newText in
            userContext = newText
        }
    }
}

struct ContentView: View {
    // State to hold the detected emotion
    @State private var emotion = "Unknown"
    @State private var responseText = ""
    @State private var showContextInput = false
    @State private var userContext = ""
    
    // Instance of HealthKit store
    let healthStore = HKHealthStore()

    // Instance of the SpeechRecognizer
    @StateObject private var speechRecognizer = SpeechRecognizer()

    // Instance of AVSpeechSynthesizer for text-to-speech
    private let speechSynthesizer = AVSpeechSynthesizer()

// Function to request authorization to access HRV data from HealthKit
    func requestHealthDataPermission() {
        guard let hrvType = HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
            print("HRV type is not available on this device.")
            return
        }
        
        healthStore.requestAuthorization(toShare: [], read: [hrvType]) { (success, error) in
            if success {
                print("Health data permission granted for HRV")
            } else {
                print("Health data permission denied: \(String(describing: error))")
            }
        }
    }


//    func fetchHealthData() {
//        guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
//            print("HRV type is not available on this device.")
//            return
//        }
//        
//        let calendar = Calendar.current
//        let now = Date()
//        let startOfDay = calendar.startOfDay(for: now)
//        let sixAM = calendar.date(bySettingHour: 6, minute: 0, second: 0, of: startOfDay)!
//        let sixPM = calendar.date(bySettingHour: 18, minute: 0, second: 0, of: startOfDay)!
//        
//        let predicate = HKQuery.predicateForSamples(withStart: sixAM, end: sixPM)
//        
//        // Fetch HRV Data Over a 12-Hour Period and Calculate Hourly Averages
//        var hourlyHRVValues: [Date: [Double]] = [:]
//        
//        let hrvQuery = HKSampleQuery(sampleType: hrvType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, results, error in
//            guard let samples = results as? [HKQuantitySample], error == nil else {
//                print("Error fetching HRV data: \(error?.localizedDescription ?? "Unknown error")")
//                return
//            }
//
//            // Grouping samples by hour and calculating hourly averages
//            for sample in samples {
//                let hour = calendar.component(.hour, from: sample.startDate)
//                let value = sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
//                let timestamp = calendar.date(bySettingHour: hour, minute: 0, second: 0, of: sample.startDate)!
//                
//                if hourlyHRVValues[timestamp] != nil {
//                    hourlyHRVValues[timestamp]?.append(value)
//                } else {
//                    hourlyHRVValues[timestamp] = [value]
//                }
//            }
//
//            // Calculate hourly averages
//            var hourlyAverages: [Date: Double] = [:]
//            for (timestamp, values) in hourlyHRVValues {
//                hourlyAverages[timestamp] = values.reduce(0, +) / Double(values.count)
//            }
//
//            // Classify emotion based on the HRV values
//            let emotionMapping: [ClosedRange<Double>: String] = [
//                0...29: "Stressed",
//                30...59: "Neutral",
//                60...100: "Relaxed"
//            ]
//            
//            var emotionCounts: [String: Int] = [:]
//            
//            // Classify each hourly HRV value into an emotion
//            for (_, hrv) in hourlyAverages {
//                let emotion = emotionMapping.first { $0.key.contains(hrv) }?.value ?? "Unknown"
//                emotionCounts[emotion, default: 0] += 1
//            }
//            
//            // Determine the dominant emotion based on frequency
//            let uniqueCountValues = Set(emotionCounts.values)
//            let finalEmotion: String
//            if uniqueCountValues.count == 1 {
//                // If all emotions have the same count, default to Neutral
//                finalEmotion = "Neutral"
//            } else {
//                finalEmotion = emotionCounts.max { $0.value < $1.value }?.key ?? "Neutral"
//            }
//
//            // Update the UI or perform any other action with the detected emotion
//            DispatchQueue.main.async {
//                print("Dominant Emotion: \(finalEmotion)")
//                self.updateEmotionBasedOnDetectedEmotion(emotion: finalEmotion)
//            }
//        }
//        
//        healthStore.execute(hrvQuery)
//    }
//
//    // Function to update emotion based on detected dominant emotion
//    func updateEmotionBasedOnDetectedEmotion(emotion: String) {
//        // Use the detected emotion (e.g., "Stressed", "Neutral", "Happy") for further action
//        print("Detected Emotion: \(emotion)")
//        // Update UI or trigger other actions based on the detected emotion
//    }





    // Fetching HRV (Heart Rate Variability) data
    func fetchHealthData() {
        guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
            print("HRV type is not available on this device.")
            return
        }
        
        let predicate = HKQuery.predicateForSamples(withStart: Date().addingTimeInterval(-60*60*24), end: Date())
        
        let hrvQuery = HKStatisticsQuery(quantityType: hrvType, quantitySamplePredicate: predicate, options: .discreteAverage) { (query, result, error) in
            if let error = error {
                print("Error fetching HRV: \(error.localizedDescription)")
                return
            }
            let hrvValue = result?.averageQuantity()?.doubleValue(for: HKUnit.secondUnit(with: .milli)) ?? 0
            print("HRV Value: \(hrvValue) ms")

            updateEmotionBasedOnHRV(hrv: hrvValue)
        }
        
        healthStore.execute(hrvQuery)
    }

    //This version of updateEmotionBasedOnHRV() is used only for demo just to show variety of emotions
    func updateEmotionBasedOnHRV(hrv: Double) {
        let emotions = ["Stressed", "Neutral",  "Relaxed"]
        
        if let randomEmotion = emotions.randomElement() {
            emotion = randomEmotion
            print("Selected Emotion: \(randomEmotion)")
        }
    }

    
    
    // Function to map HRV (Heart Rate Variability) to emotion
//    func updateEmotionBasedOnHRV(hrv: Double) {
//
//        if hrv < 30 {
//            emotion = "Stressed"
//        } else if hrv < 50 {
//            emotion = "Neutral"
//        } else {
//            emotion = "Relaxed"
//        }
        // Only call API after emotion is updated, not here during data fetching
//    }

    // Function to call FastAPI server and fetch the response text
    func callAPI(emotion: String, userContext: String, top_k: Int) {
        // Make sure context is not empty before calling API
        if userContext.isEmpty {
            print("Context is empty, API call will not be made.")
            return
        }
                
        print("top_k: \(top_k)")
        let url = URL(string: "http://192.168.1.135:8000/search?word=\(emotion)&userContext=\(userContext)&top_k=\(top_k)")!
        //let url = URL(string: //"http://ec2-54-152-218-208.compute-1.amazonaws.com:8000/search?word=\(emotion)&userContext=\(userContext)&top_k=\(top_k)")!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        print("emotion: ")
        print(emotion)
        print("userContext:")
        print(userContext)
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Error calling API: \(error)")
                return
            }
            
            //print raw string for debug
            if let data = data, let rawString = String(data: data, encoding: .utf8) {
                    print("Raw Response:", rawString)  // Check if JSON is malformed
                }
            
            
            
            guard let data = data else {
                print("No data received")
                return
            }
            
            do {
                print("data: \(data)")
                let jsonResponse = try JSONSerialization.jsonObject(with: data, options: [])
                print("jsonResponse: \(jsonResponse)")
                if let responseDict = jsonResponse as? [String: Any] {
                    // Expecting 'results' to be a string, not an array of dictionaries
                    print("responseDict: \(responseDict)")
                    if let results = responseDict["results"] as? String {
                        DispatchQueue.main.async {
                            // Use the string directly
                            self.responseText = results
                            self.speakResponseText(results)  // Convert text to speech
                        }
                    }
                }
            } catch {
                print("Error decoding response: \(error)")
            }
        }

        task.resume()
    }

    // Function to speak out the response text
    func speakResponseText(_ text: String) {
        //let utterance = AVSpeechUtterance(string: text)
        //utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        //speechSynthesizer.speak(utterance)
    }

    // Chakra Area (e.g., Navel region) click handler
    func chakraClickHandler(chakraName: String) {
        self.showContextInput = true // Show the speech-to-text view
        print("showContextInput:")
        print(showContextInput)
    }

    var body: some View {
        VStack {
            Text("Detected Emotion: \(emotion)")
                .font(.largeTitle)
                .padding()

            if emotion == "Stressed" || emotion == "Neutral" || emotion == "Relaxed" {
                Image("stressedImage") // Replace with your actual image name
                    .resizable()
                    .scaledToFit()
                    .frame(width: 200, height: 200)
                    .overlay(
                        // Example: Highlighting navel area with a tap gesture
                        Circle()
                            .stroke(Color.red, lineWidth: 3)
                            .frame(width: 50, height: 50)
                            .position(x: 100, y: 125) // Adjust position for the navel area
                            .onTapGesture {
                                chakraClickHandler(chakraName: "Navel")
                            }
                    )
            }

            if showContextInput {
                
                SpeechToTextView(userContext: $userContext)
                    

                Button("Submit") {
                    // Ensure user context is non-empty before calling API
                    print("touched Navel")
                    if !userContext.isEmpty {
                        print("userContext")
                        print(userContext)
                        callAPI(emotion: emotion, userContext: userContext, top_k: 3)
                    } else {
                        print("No context entered. API call aborted.")
                    }
                    showContextInput = false // Hide the speech-to-text view
                }
                .padding()
            }

            ScrollView {
                Text(responseText)
                    .font(.title)
                    .padding()
                    .frame(maxWidth: .infinity)
            }
            .frame(height: 300)

            Button("Fetch Health Data") {
                requestHealthDataPermission()
                fetchHealthData()
            }
            .padding()
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

