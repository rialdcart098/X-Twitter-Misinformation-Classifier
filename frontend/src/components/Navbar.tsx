import githubLogo from '../assets/github-logo.png'

function Navbar(){
    return (
        <nav className='bg-purple-900 justify-between w-full p-4 grid grid-cols-[0.75fr_0.25fr] place-items-center border-b-2 border-b-purple-400'>
            <h1 className=' text-2xl text-green-300 font-medium font-almarai'>
                X/Twitter Misinformation Classifier
            </h1>
            <a href="https://github.com/rialdcart098/X-Twitter-Misinformation-Classifier">
                <img src={githubLogo} alt="Source Code" className='w-8 h-8 hover:drop-shadow-[0_0_10px_rgba(191,219,254,1)] rounded-2xl transition-all ease-in-out'/>
            </a>
        </nav>
    )
}
export default Navbar