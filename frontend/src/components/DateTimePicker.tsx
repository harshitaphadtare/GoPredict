import React, { useState, useEffect } from "react";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import "./DateTimePicker.css";
import { Calendar, Clock} from "lucide-react";

interface DateTimePickerProps {
  selected: Date | null;
  onChange: (date: Date | null) => void;
}

const CustomInput = React.forwardRef<
  HTMLDivElement,
  {
    value?: string;
    onClick?: () => void;
    icon: React.ReactNode;
    placeholder: string;
  }
>(({ value, onClick, icon, placeholder }, ref) => (
  <div
    ref={ref}
    onClick={onClick}
    className="cursor-pointer rounded-lg border border-border bg-background px-3 py-2 text-foreground shadow-soft transition focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/30 flex justify-between items-center w-full"
  >
    <span className="text-sm">{value || placeholder}</span>
    {icon}
  </div>
));

export const DateTimePicker = ({ selected, onChange }: DateTimePickerProps) => {
  const [isMobile, setIsMobile] = useState(false);
  const [date, setDate] = useState<Date | null>(null);
  const [time, setTime] = useState<Date | null>(null);

  useEffect(() => {
    setDate(selected);
    setTime(selected);
  }, [selected]);

  useEffect(() => {
    const checkIsMobile = () => setIsMobile(window.innerWidth <= 768);
    checkIsMobile(); // Initial check
    window.addEventListener("resize", checkIsMobile);
    return () => window.removeEventListener("resize", checkIsMobile);
  }, []);

  const handleDateChange = (newDate: Date | null) => {
    if (newDate) {
      const newDateTime = new Date(newDate);
      if (time) {
        newDateTime.setHours(time.getHours(), time.getMinutes());
      }
      onChange(newDateTime < new Date() ? new Date() : newDateTime);
    } else {
      onChange(null);
    }
  };

  const handleTimeChange = (newTime: Date | null) => {
    if (newTime) {
      // Use the existing date, or default to today if no date is selected
      const newDateTime = date ? new Date(date) : new Date();
      newDateTime.setHours(newTime.getHours(), newTime.getMinutes());
      onChange(newDateTime < new Date() ? new Date() : newDateTime); // Ensure time is not in the past
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
    <div className="grid grid-cols-2 gap-2">
      <div className="grid">
        <DatePicker
          selected={date}
          onChange={handleDateChange}
          withPortal={isMobile}
          dateFormat="dd-MM-yyyy"
          popperPlacement={(isMobile ? "auto" : "bottom-start") as any}
          minDate={new Date()}
          showPopperArrow={false}
          customInput={
            <CustomInput
              placeholder="Select Date"
              icon={<Calendar className="h-4 w-4 text-foreground/60" />}
            />
          }
        />
      </div>
      <div className="grid">
        <DatePicker
          selected={time}
          onChange={handleTimeChange}
          withPortal={isMobile}
          showTimeSelect
          popperPlacement={(isMobile ? "auto" : "bottom-start") as any}
          showTimeSelectOnly
          timeIntervals={5}
          timeCaption="Time"
          dateFormat="HH:mm"
          minTime={isSameDay(date, new Date()) ? new Date() : undefined}
          maxTime={isSameDay(date, new Date()) ? new Date(new Date().setHours(23, 59, 59)) : undefined}
          showPopperArrow={false}
          customInput={
            <CustomInput
              placeholder="Select Time"
              icon={<Clock className="h-4 w-4 text-foreground/60" />}
            />
          }
        />
      </div>
    </div>
  );
};