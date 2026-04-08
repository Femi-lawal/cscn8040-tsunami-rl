import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "Tsunami Warning RL Console",
    description: "Operations console for tsunami early warning RL decision support",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="en">
            <head>
                <link
                    href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css"
                    rel="stylesheet"
                />
            </head>
            <body>{children}</body>
        </html>
    );
}
