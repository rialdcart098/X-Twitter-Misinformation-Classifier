import Navbar from './components/Navbar.tsx'
import Footer from './components/Footer.tsx'
import Content from './components/Content.tsx'

function App() {
    return (
        <div className="grid h-screen bg-purple-950 grid-rows-[0.25fr_2fr_0.5fr]">
            <Navbar />
            <Content />
            <Footer />
        </div>
    )
}

export default App
