import { motion } from 'motion/react'
import { Button } from '../components/ui/button'

export function Cover({ onOpen }: { onOpen: () => void }) {
  return (
    <motion.main
      initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
      className="mx-auto grid min-h-screen max-w-5xl content-center gap-10 px-6 py-12 md:grid-cols-[1.3fr_1fr] md:items-center"
    >
      <div>
        <h1 className="font-display text-7xl font-bold leading-none sm:text-8xl">
          death<span className="text-blood">note</span>
        </h1>
        <p className="mt-6 max-w-md font-display text-3xl leading-tight">
          How long is your page, and which habits are writing it shorter?
        </p>
        <p className="mt-4 max-w-md text-paper/70">
          Answer four short pages about your age, sleep, habits and health. You get an estimated lifespan range set
          against Indian death rates, and can try changes to see what moves it.
        </p>
      </div>
      <div className="sheet py-10 pl-14 pr-8">
        <h2 className="font-display text-3xl font-bold">Before you open it</h2>
        <ul className="mt-4 space-y-3">
          <li>An estimate for people like you, not a prediction about you.</li>
          <li>Not medical advice. A doctor can judge your health; this cannot.</li>
          <li>Your answers are not saved anywhere.</li>
          <li>For ages 20 to 79.</li>
        </ul>
        <Button variant="blood" className="mt-8 w-full" onClick={onOpen}>Open the notebook</Button>
      </div>
    </motion.main>
  )
}
