// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use iced::widget::{container, progress_bar, text, tooltip};
use iced::{Element, Length, Theme, color};

use crate::gui::Message;
use crate::gui::theme::vu_color;

/// Create a VU meter bar for a single channel side.
pub fn vu_bar(value: f32) -> Element<'static, Message> {
    let clamped = value.clamp(0.0, 1.0);
    let color = vu_color(clamped);
    container(
        progress_bar(0.0..=1.0, clamped)
            .height(6)
            .style(move |_theme: &iced::Theme| progress_bar::Style {
                background: iced::Background::Color(iced::Color::from_rgba8(
                    0x1a, 0x1a, 0x2a, 0.75,
                )),
                bar: iced::Background::Color(color),
                border: iced::Border::default(),
            }),
    )
    .width(Length::Fill)
    .into()
}

fn tooltip_content<'a>(message: impl Into<String>) -> Element<'a, Message> {
    container(
        text(message.into())
            .size(11)
            .color(color!(0xd8, 0xf4, 0xff)),
    )
    .padding([4, 8])
    .style(|_theme: &Theme| container::Style {
        background: Some(iced::Background::Color(color!(0x08, 0x12, 0x18))),
        border: iced::Border {
            color: color!(0x00, 0xcc, 0xff),
            width: 1.0,
            radius: 6.0.into(),
        },
        ..Default::default()
    })
    .into()
}

pub fn with_tooltip<'a>(
    content: impl Into<Element<'a, Message>>,
    message: impl Into<String>,
) -> Element<'a, Message> {
    with_tooltip_at(content, message, tooltip::Position::Bottom)
}

pub fn with_tooltip_at<'a>(
    content: impl Into<Element<'a, Message>>,
    message: impl Into<String>,
    position: tooltip::Position,
) -> Element<'a, Message> {
    tooltip(content, tooltip_content(message), position)
        .gap(6)
        .padding(0)
        .into()
}

/// Format seconds as MM:SS.
pub fn format_time(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    let mins = total / 60;
    let secs = total % 60;
    format!("{mins:02}:{secs:02}")
}

/// Format order/row display.
pub fn format_position(order: i32, row: i32) -> String {
    format!("Ord:{order:03} Row:{row:03}")
}
