import { useLocation, Outlet } from "react-router-dom";
import { AnimatePresence, motion } from "motion/react";
import { TopNav } from "@/components/app/TopNav";

const PAGE_VARIANTS = {
  initial: { opacity: 0, y: 6 },
  animate: { opacity: 1, y: 0 },
  exit:    { opacity: 0, y: -4 },
};

const PAGE_TRANSITION = { duration: 0.28, ease: [0.16, 1, 0.3, 1] };

export default function Layout() {
  const location = useLocation();

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <TopNav />
      <main className="flex-1 pt-[56px]">
        <div className="max-w-[1440px] mx-auto px-6 py-8">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              variants={PAGE_VARIANTS}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={PAGE_TRANSITION}
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
