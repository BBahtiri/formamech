import type { FrontendRenderer, FrontendRendererArgs } from "@streamlit/component-v2-lib";
import { createRoot, type Root } from "react-dom/client";
import { Viewer3D } from "./Viewer3D";
import type { ApplyFaceEvent3D, Viewer3DData } from "./types";
import "./styles.css";

const roots = new WeakMap<FrontendRendererArgs["parentElement"], Root>();

const mount: FrontendRenderer<{ apply: ApplyFaceEvent3D }, Viewer3DData> = ({
  data,
  parentElement,
  setTriggerValue,
}) => {
  const element = parentElement.querySelector(".react-root");
  if (!element) throw new Error("3D Viewer mount element is missing.");

  let root = roots.get(parentElement);
  if (!root) {
    root = createRoot(element);
    roots.set(parentElement, root);
  }
  root.render(
    <Viewer3D data={data} onApply={(event) => setTriggerValue("apply", event)} />,
  );

  return () => {
    roots.get(parentElement)?.unmount();
    roots.delete(parentElement);
  };
};

export default mount;
