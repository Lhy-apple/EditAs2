/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:45:02 GMT 2023
 */

package com.fasterxml.jackson.dataformat.xml.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.ctc.wstx.stax.WstxInputFactory;
import com.fasterxml.jackson.dataformat.xml.deser.XmlTokenStream;
import java.io.Reader;
import java.io.StringReader;
import javax.xml.stream.XMLStreamReader;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class XmlTokenStream_ESTest extends XmlTokenStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      WstxInputFactory wstxInputFactory0 = new WstxInputFactory();
      StringReader stringReader0 = new StringReader("org.codehaus.stax2.closeInputSource");
      XMLStreamReader xMLStreamReader0 = wstxInputFactory0.createXMLStreamReader("com.ctc.wstx.lazyParsing", (Reader) stringReader0);
      XmlTokenStream xmlTokenStream0 = null;
      try {
        xmlTokenStream0 = new XmlTokenStream(xMLStreamReader0, stringReader0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid XMLStreamReader passed: should be pointing to START_ELEMENT (1), instead got 7
         //
         verifyException("com.fasterxml.jackson.dataformat.xml.deser.XmlTokenStream", e);
      }
  }
}