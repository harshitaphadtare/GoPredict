import React from "react";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import "./DateTimePicker.css"; // Import custom styles
import { Calendar, X } from "lucide-react";

interface DateTimePickerProps {
  selected: Date | null;
  onChange: (date: Date | null) => void;
}

const CustomInput = React.forwardRef<
  HTMLDivElement,
  { value?: string; onClick?: () => void; onClear: () => void; hasValue: boolean }
>(({ value, onClick, onClear, hasValue }, ref) => (
  <div
    ref={ref}
    onClick={onClick}
    className="cursor-pointer rounded-lg border border-border bg-background px-4 py-3 text-foreground shadow-soft transition focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/30 flex justify-between items-center"
  >
    <span className="text-sm">{value || "dd-mm-yyyy --:--"}</span>
    <div className="flex items-center gap-2">
      {hasValue && (
        <button
          type="button"
          className="p-1 rounded-full hover:bg-muted"
          onClick={(e) => {
            e.stopPropagation(); // Prevent opening the picker
            onClear();
          }}
        >
          <X className="h-3 w-3 text-foreground/60" />
        </button>
      )}
      <Calendar className="h-4 w-4 text-foreground/60" />
    </div>
  </div>
));

export const DateTimePicker = ({ selected, onChange }: DateTimePickerProps) => {
  const handleDateChange = (date: Date | null) => {
    // Ensure selected time is not in the past
    if (date && date < new Date()) {
      onChange(new Date());
    } else {
      onChange(date);
    }
  };
  const isSameDay = (d1: Date | null, d2: Date | null) => {
    if (!d1 || !d2) return false;
    return (
      d1.getFullYear() === d2.getFullYear() &&
      d1.getMonth() === d2.getMonth() &&
      d1.getDate() === d2.getDate()
    );
  };

  return (
    <div className="grid grid-cols-1">
      <DatePicker
        selected={selected}
        onChange={handleDateChange}
        showTimeSelect
        dateFormat="dd-MM-yyyy HH:mm"
        timeIntervals={1}
        minDate={new Date()}
        minTime={isSameDay(selected, new Date()) ? new Date() : undefined}
        maxTime={isSameDay(selected, new Date()) ? new Date(new Date().setHours(23, 59, 59)) : undefined}
        showPopperArrow={false}
        customInput={
          <CustomInput
            hasValue={!!selected}
            onClear={() => onChange(null)}
          />
        }
      />
    </div>
  );
};