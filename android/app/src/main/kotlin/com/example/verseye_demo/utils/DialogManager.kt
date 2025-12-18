package com.example.verseye_demo.utils

import android.content.Context
import androidx.appcompat.app.AlertDialog

/**
 * DialogManager handles all dialog operations (loading, error, info dialogs)
 */
class DialogManager(private val context: Context) {
    
    private var loadingDialog: AlertDialog? = null
    
    /**
     * Show a loading dialog with a message
     */
    fun showLoadingDialog(message: String) {
        dismissLoadingDialog()
        
        val builder = AlertDialog.Builder(context)
        builder.setTitle("Loading")
        builder.setMessage(message)
        builder.setCancelable(false)
        loadingDialog = builder.create()
        loadingDialog?.show()
    }
    
    /**
     * Update the loading dialog message
     */
    fun updateLoadingDialog(message: String) {
        loadingDialog?.setMessage(message)
    }
    
    /**
     * Dismiss the loading dialog with optional final message
     */
    fun dismissLoadingDialog(finalMessage: String? = null) {
        finalMessage?.let {
            loadingDialog?.setMessage(it)
        }
        
        loadingDialog?.dismiss()
        loadingDialog = null
    }
    
    /**
     * Show an error dialog
     */
    fun showErrorDialog(message: String, onDismiss: (() -> Unit)? = null) {
        AlertDialog.Builder(context)
            .setTitle("Error")
            .setMessage(message)
            .setPositiveButton("OK") { _, _ ->
                onDismiss?.invoke()
            }
            .show()
    }
    
    /**
     * Show an info dialog
     */
    fun showInfoDialog(title: String, message: String, onDismiss: (() -> Unit)? = null) {
        AlertDialog.Builder(context)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("OK") { _, _ ->
                onDismiss?.invoke()
            }
            .show()
    }
    
    /**
     * Show a confirmation dialog
     */
    fun showConfirmationDialog(
        title: String, 
        message: String, 
        onConfirm: () -> Unit, 
        onCancel: (() -> Unit)? = null
    ) {
        AlertDialog.Builder(context)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("Yes") { _, _ ->
                onConfirm()
            }
            .setNegativeButton("No") { _, _ ->
                onCancel?.invoke()
            }
            .show()
    }
    
    /**
     * Check if loading dialog is showing
     */
    fun isLoadingDialogShowing(): Boolean {
        return loadingDialog?.isShowing == true
    }
}
