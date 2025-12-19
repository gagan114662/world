import { useEffect, useRef, memo } from "react";
import vegaEmbed from "vega-embed";

interface AltairProps {
  json_graph: string;
}

function AltairComponent({ json_graph }: AltairProps) {
  const embedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (embedRef.current && json_graph) {
      try {
        const spec = JSON.parse(json_graph);
        vegaEmbed(embedRef.current, spec, {
          actions: false,
          responsive: true,
          width: 'container'
        });
      } catch (e) {
        console.error("Failed to parse Altair graph", e);
      }
    }
  }, [embedRef, json_graph]);

  return <div className="vega-embed w-full h-[300px]" ref={embedRef} />;
}

export const Altair = memo(AltairComponent);
