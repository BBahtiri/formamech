import { useEffect, useId, useState } from "react";
import { BoundaryEditor, boundaryDraft, draftAssignment, type BoundaryDraft } from "./BoundaryEditor";
import { BoundaryPlot } from "./BoundaryPlot";
import { KIND_LABELS, type ApplyEvent, type ApplyResult, type Assignment, type PickerData } from "./types";

type Drafts = Record<string, BoundaryDraft>;
type Submission = { event: ApplyEvent; draft?: BoundaryDraft };

function appliedValues(assignment: Assignment) {
  return assignment.components?.map((component, index) => {
    const value = assignment.values?.[index];
    return `u_${component} = ${value?.expression ?? "0"} ${value?.unit ?? "mm"}`;
  }).join(", ") || "—";
}

function reconcileDrafts(drafts: Drafts, submission: Submission, result: ApplyResult, data: PickerData) {
  const { event, draft } = submission;
  const updated = { ...drafts };
  if (updated[event.region] === draft) delete updated[event.region];
  if (data.geometry_id === event.geometry_id && data.draft_revision === result.draft_revision) {
    for (const [region, other] of Object.entries(updated)) {
      if (other.geometry_id === event.geometry_id && other.draft_revision === event.draft_revision) {
        updated[region] = { ...other, draft_revision: data.draft_revision };
      }
    }
  }
  return updated;
}

export function BoundaryPicker({ data, onApply }: { data: PickerData; onApply: (event: ApplyEvent) => void }) {
  const [selection, setSelection] = useState("");
  const [drafts, setDrafts] = useState<Drafts>({});
  const [submission, setSubmission] = useState<Submission | null>(null);
  const [feedback, setFeedback] = useState<ApplyResult | null>(null);
  const [notice, setNotice] = useState("");
  const id = useId();

  useEffect(() => {
    const result = data.last_apply;
    if (!result || !submission || result.event_id !== submission.event.event_id) return;
    if (result.status === "accepted") {
      setDrafts((previous) => reconcileDrafts(previous, submission, result, data));
    }
    setFeedback(result);
    setSubmission(null);
  }, [data, submission]);

  const accepted = data.assignments.find((entry) => entry.region === selection) ?? data.assignments[0];
  if (!accepted) return <p>No boundary groups are available.</p>;

  const region = accepted.region;
  const pending = drafts[region];
  const draft = pending ?? boundaryDraft(accepted, data);
  const errors = feedback?.region === region ? feedback.errors : undefined;
  const reloadRequired = feedback?.region === region && feedback.reload === true;

  function edit(fields: Partial<BoundaryDraft>) {
    setDrafts((previous) => ({ ...previous, [region]: { ...draft, ...fields } }));
  }

  function submit() {
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    const eventId = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
    const event: ApplyEvent = {
      workspace_id: data.workspace_id,
      geometry_id: draft.geometry_id,
      draft_revision: draft.draft_revision,
      event_id: eventId,
      region,
      assignment: pending ? draftAssignment(region, pending) : accepted,
    };
    setNotice("");
    setFeedback(null);
    setSubmission({ event, draft: pending });
    onApply(event);
  }

  function reload() {
    if (pending) edit({ geometry_id: data.geometry_id, draft_revision: data.draft_revision });
    setFeedback(null);
    setNotice("Accepted conditions loaded. Review the current assignment and your draft before applying.");
  }

  const message = feedback?.message ?? (feedback?.status === "invalid" ? "Check the boundary condition fields." : "");

  return (
    <div className="boundary-picker">
      <div className="boundary-field boundary-selector">
        <label htmlFor={`${id}-region`}>Selected CAD Boundary Group</label>
        <select id={`${id}-region`} value={region} onChange={(event) => setSelection(event.target.value)}
          aria-invalid={errors?.region ? true : undefined} aria-describedby={errors?.region ? `${id}-region-error` : undefined}>
          {data.assignments.map((assignment) => <option key={assignment.region} value={assignment.region}>
            {assignment.region} · {assignment.name} · {KIND_LABELS[assignment.kind]}
          </option>)}
        </select>
        {errors?.region && <p id={`${id}-region-error`} className="boundary-field-error">{errors.region}</p>}
      </div>
      <div className="boundary-grid">
        <BoundaryPlot geometry_id={data.geometry_id} draft_revision={data.draft_revision} figure={data.figure}
          mesh={data.preview.mesh} selected={region} onSelect={setSelection} />
        <div className="boundary-controls">
          <BoundaryEditor region={region} draft={draft} pending={Boolean(pending)}
            suggested={data.suggested_regions.includes(region)} errors={errors} onEdit={edit} />
          <div className="boundary-actions">
            <button className="boundary-apply" type="button" disabled={Boolean(submission) || reloadRequired}
              onClick={submit}>Apply boundary condition</button>
            {reloadRequired && <button type="button" onClick={reload}>Reload accepted conditions</button>}
          </div>
          <p role="status" className="boundary-status">{submission ? "Applying boundary condition…" : errors?.form ?? (message || notice)}</p>
        </div>
      </div>
      {data.readiness_errors.length > 0 && <div className="boundary-readiness">
        <h3>Study readiness</h3>
        <ul>{data.readiness_errors.map((error, index) => <li key={index}>{error}</li>)}</ul>
      </div>}
      <div className="boundary-table" role="region" aria-label="Accepted boundary assignments" tabIndex={0}>
        <table>
          <caption>Accepted boundary assignments</caption>
          <thead><tr><th scope="col">Edge</th><th scope="col">Condition</th><th scope="col">Type</th><th scope="col">Displacement</th></tr></thead>
          <tbody>{data.assignments.map((assignment) => (
            <tr key={assignment.region} aria-current={assignment.region === region ? true : undefined}>
              <th scope="row">{assignment.region}</th>
              <td>{assignment.name}{data.suggested_regions.includes(assignment.region) && <small>Suggested from geometry</small>}</td>
              <td>{KIND_LABELS[assignment.kind]}</td>
              <td>{appliedValues(assignment)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  );
}
