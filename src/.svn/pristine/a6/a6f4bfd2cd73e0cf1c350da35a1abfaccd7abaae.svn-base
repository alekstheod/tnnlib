/**
*  Copyright (c) 2011, Alex Theodoridis
*  All rights reserved.

*  Redistribution and use in source and binary forms, with 
*  or without modification, are permitted provided that the 
*  following conditions are met:
*  Redistributions of source code must retain the above 
*  copyright notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above 
*  copyright notice, this list of conditions and the following
*  disclaimer in the documentation and/or other materials 
*  provided with the distribution.

*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
*  AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
*  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
*  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
*  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
*  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
*  OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
*  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
*  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
*  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE,
*  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
*/


#ifndef UTILS_AEXCEPTION_H
#define UTILS_AEXCEPTION_H

#include <exception>
#include <string>

/// <summary>
/// The main utilities namespace that contain the utilities classes.
/// </summary>
namespace utils {

    /// <summary>
    /// The basic exception class
    /// </summary>
    class AException : public std::exception {
    private:
        /// <summary>
        /// Exception message
        /// </summary>
        std::string m_message;

    public:
        /// <summary>
        /// Basic exception constructor
        /// </summary>
        /// <param name="message">The exception message</param>
        /// <param name="sysMessage">Boolean argument that can allow to append the system message, false as default</sysMessage>
        AException(const std::string &message, bool sysMessage = false);

        /// <summary>
        /// Method that return the pointer to message string.
        /// </summary>
        /// <returns>Pointer to message string</returns>
        virtual const char* what(void) const throw ();

        /// <summary>
        /// Destructor.
        /// </summary>
        virtual ~AException(void) throw ();
    };

}

#endif

