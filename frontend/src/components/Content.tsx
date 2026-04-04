import axios from "axios";
import { useState } from "react";
import thumbsUp from '../assets/thumbs-up.svg'
import thumbsDown from '../assets/thumbs-down.svg'
import * as React from "react";

function Content() {
    const BACKEND_URL = import.meta.env.VITE_BACKEND_URL

    const [prediction, setPrediction] = useState<string | null>(null);
    const [confidence, setConfidence] = useState<number | null>(null);
    const [tweet_text, setTweetText] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [feedback, setFeedback] = useState<boolean | null>(null);

    async function classify(event: React.FormEvent<HTMLFormElement>) {
        event.preventDefault()
        setLoading(true)
        const form = event.target as HTMLFormElement
        const linkInput = form.elements.namedItem('message') as HTMLTextAreaElement
        const link = linkInput.value
        axios.post(`${BACKEND_URL}/predict`, { tweet: link })
            .then(response => {
                setPrediction(response.data.prediction);
                setConfidence(response.data.confidence);
                setTweetText(response.data.tweet)
                setLoading(false)
            })
            .catch(err => {
                console.error('Error classifying tweet:', err)
                alert('An error occurred while classifying the tweet. Please try again later.')
            })
    }

    async function send_feedback(event: React.FormEvent<HTMLFormElement>) {
        event.preventDefault()
        if (feedback === null) {
            alert('Please select feedback.')
        }
        const feedback_prediction = feedback ? prediction : !prediction
        axios.post(`${BACKEND_URL}/feedback`, {
            tweet_text,
            prediction: feedback_prediction
        }).then(response => {
            alert(response.data.message)
        }).catch(error => {
            console.error('Error sending feedback:', error)
            alert('An error occurred while sending feedback. Please try again later.')
        })
    }

    return (
        <div className='flex flex-col items-center justify-center gap-8'>
            <h2 className='text-3xl text-green-300 font-almarai font-medium'>Enter the tweet you want to classify:</h2>
            <form 
                onSubmit={classify}
                className='flex items-center'
            >
                <textarea
                    name='message'
                    placeholder='Enter tweet text here'
                    className='py-1.5 pr-3 pl-1 p-2 rounded w-96 bg-purple-200 focus:outline-none'
                    onInput={(e) => {
                        (e.target as HTMLTextAreaElement).style.height = "auto";
                        (e.target as HTMLTextAreaElement).style.height = (e.target as HTMLTextAreaElement).scrollHeight + "px";
                    }}
                />
                <button type='submit' className='cursor-pointer m-4 bg-green-300 p-3 font-semibold text-xl text-purple-950 font-almarai rounded-lg hover:text-purple-800 transition-all ease-in-out'>Classify</button>
            </form>
            {prediction === null && loading &&
                <p className='text-green-400 font-almarai font-thin'>Classifying... (This can take a while)</p>
            }
            {prediction !== null && confidence !== null && (
                <div className='text-center'>
                    <h3 className='text-2xl text-green-300 font-almarai font-medium'>Our model predicted this post as</h3>
                    <h3 className={`text-4xl ${prediction ? 'text-green-400' : 'text-red-400'} font-almarai font-bold`}>
                        {prediction ? 'LEGITIMATE' : 'MISINFORMATION'}
                    </h3>
                    <p className='text-green-300 font-almarai font-thin'>Confidence: {confidence * 100}%</p>
                    <form onSubmit={send_feedback}>
                        <h3 className='font-bold text-green-300 font-almarai mt-4'>
                            Feedback: (Help us train our model!)
                        </h3>
                        <div className='flex flex-row justify-center gap-12 mt-4'>
                            <label className={`cursor-pointer ${feedback === true ? 'bg-green-400' : ''} rounded-4xl p-2 transition ease-in-out`}>
                                <input
                                    type="radio"
                                    name="feedback"
                                    value="correct"
                                    className="hidden"
                                    onChange={() => setFeedback(true)}
                                />
                                <img
                                    src={thumbsUp}
                                    alt="Correct"
                                    className="w-12 h-12 rounded-xl transition transform hover:scale-110 hover:shadow-lg"
                                />
                            </label>
                            <label className={`cursor-pointer ${feedback === false ? 'bg-red-400' : ''} rounded-4xl p-2 transition ease-in-out`}>
                                <input
                                    type="radio"
                                    name="feedback"
                                    value="incorrect"
                                    className="hidden"
                                    onChange={() => setFeedback(false)}
                                />
                                <img
                                    src={thumbsDown}
                                    alt="Incorrect"
                                    className="w-12 h-12 rounded-xl transition transform hover:scale-110 hover:shadow-lg "
                                />
                            </label>
                            <button type='submit' className='cursor-pointer font-almarai font-bold bg-green-400 rounded-xl p-2 hover:bg-green-500 transition ease-in-out'>Submit Feedback</button>
                        </div>
                    </form>
                </div>
            )}
        </div>
    )
}
export default Content;
