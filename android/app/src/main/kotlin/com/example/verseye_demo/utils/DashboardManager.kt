package com.example.verseye_demo.utils

import android.content.Context
import android.graphics.Color
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog

/**
 * DashboardManager handles the dashboard UI for displaying object detection counts
 */
class DashboardManager(private val context: Context) {
    
    companion object {
        private val LABEL_TO_EMOJI: Map<String, String> = mapOf(
            "person" to "🧍", "bicycle" to "🚲", "car" to "🚗", "motorcycle" to "🏍️",
            "airplane" to "✈️", "bus" to "🚌", "train" to "🚆", "truck" to "🚚",
            "boat" to "🚤", "traffic light" to "🚦", "fire hydrant" to "🚒",
            "stop sign" to "🛑", "parking meter" to "🅿️", "bench" to "🪑",
            "bird" to "🐦", "cat" to "🐱", "dog" to "🐶", "horse" to "🐴",
            "sheep" to "🐑", "cow" to "🐄", "elephant" to "🐘", "bear" to "🐻",
            "zebra" to "🦓", "giraffe" to "🦒", "backpack" to "🎒", "umbrella" to "☂️",
            "handbag" to "👜", "tie" to "👔", "suitcase" to "🧳", "frisbee" to "🥏",
            "skis" to "🎿", "snowboard" to "🏂", "sports ball" to "⚽", "kite" to "🪁",
            "baseball bat" to "⚾", "baseball glove" to "🥎", "skateboard" to "🛹",
            "surfboard" to "🏄", "tennis racket" to "🎾", "bottle" to "🍾",
            "wine glass" to "🍷", "cup" to "☕", "fork" to "🍴", "knife" to "🔪",
            "spoon" to "🥄", "bowl" to "🥣", "banana" to "🍌", "apple" to "🍎",
            "sandwich" to "🥪", "orange" to "🍊", "broccoli" to "🥦", "carrot" to "🥕",
            "hot dog" to "🌭", "pizza" to "🍕", "donut" to "🍩", "cake" to "🎂",
            "chair" to "🪑", "couch" to "🛋️", "potted plant" to "🪴", "bed" to "🛏️",
            "dining table" to "🍽️", "toilet" to "🚽", "tv" to "📺", "laptop" to "💻",
            "mouse" to "🖱️", "remote" to "📱", "keyboard" to "⌨️", "cell phone" to "📱",
            "microwave" to "📦", "oven" to "🔥", "toaster" to "🍞", "sink" to "🚰",
            "refrigerator" to "🧊", "book" to "📖", "clock" to "🕐", "vase" to "🏺",
            "scissors" to "✂️", "teddy bear" to "🧸", "hair drier" to "💇", "toothbrush" to "🪥"
        )
    }
    
    private var dashboardDialog: AlertDialog? = null
    private var dashboardContainer: LinearLayout? = null
    private val counts: MutableMap<String, Int> = mutableMapOf()
    
    /**
     * Update detection counts
     */
    fun updateCounts(newCounts: Map<String, Int>) {
        counts.clear()
        counts.putAll(newCounts)
        updateDashboardIfVisible()
    }
    
    /**
     * Show the dashboard dialog
     */
    fun showDashboard() {
        val dialogView = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
        }

        dashboardContainer = dialogView

        val builder = AlertDialog.Builder(context)
        builder.setTitle("📊 Detection Dashboard")
        builder.setView(dialogView)
        builder.setPositiveButton("Close", null)

        dashboardDialog = builder.create()
        dashboardDialog?.show()
        updateDashboardIfVisible()
    }
    
    /**
     * Update the dashboard UI if it's visible
     */
    private fun updateDashboardIfVisible() {
        val dialog = dashboardDialog ?: return
        if (!dialog.isShowing) return
        val container = dashboardContainer ?: return
        
        container.post {
            container.removeAllViews()
            
            if (counts.isEmpty()) {
                val tv = TextView(context).apply {
                    text = "🔍 No objects detected"
                    textSize = 16f
                    gravity = Gravity.CENTER
                    setPadding(16, 32, 16, 32)
                    setTextColor(Color.GRAY)
                }
                container.addView(tv)
            } else {
                val header = TextView(context).apply {
                    text = "Live Detection (${counts.values.sum()} total)"
                    textSize = 14f
                    setTextColor(Color.GRAY)
                    setPadding(0, 0, 0, 16)
                }
                container.addView(header)
                
                counts.entries.sortedByDescending { it.value }.forEach { (label, cnt) ->
                    val row = LinearLayout(context).apply {
                        orientation = LinearLayout.HORIZONTAL
                        gravity = Gravity.CENTER_VERTICAL
                        setPadding(8, 12, 8, 12)
                    }

                    val emoji = LABEL_TO_EMOJI[label] ?: "❓"
                    
                    val emojiView = TextView(context).apply {
                        text = emoji
                        textSize = 32f
                        setPadding(0, 0, 16, 0)
                    }
                    
                    val countBadge = TextView(context).apply {
                        text = "× $cnt"
                        textSize = 20f
                        setTextColor(Color.WHITE)
                        setPadding(16, 8, 16, 8)
                        setBackgroundResource(android.R.drawable.dialog_holo_dark_frame)
                        gravity = Gravity.CENTER
                        layoutParams = LinearLayout.LayoutParams(
                            LinearLayout.LayoutParams.WRAP_CONTENT,
                            LinearLayout.LayoutParams.WRAP_CONTENT
                        )
                    }

                    row.addView(emojiView)
                    row.addView(countBadge)
                    container.addView(row)
                }
            }
        }
    }
    
    /**
     * Get emoji for a label
     */
    fun getEmojiForLabel(label: String): String {
        return LABEL_TO_EMOJI[label] ?: "❓"
    }
    
    /**
     * Dismiss the dashboard
     */
    fun dismiss() {
        dashboardDialog?.dismiss()
        dashboardDialog = null
        dashboardContainer = null
    }
}
