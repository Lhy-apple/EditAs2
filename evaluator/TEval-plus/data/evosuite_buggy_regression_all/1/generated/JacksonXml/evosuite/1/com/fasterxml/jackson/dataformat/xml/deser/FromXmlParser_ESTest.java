/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:44:24 GMT 2023
 */

package com.fasterxml.jackson.dataformat.xml.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.dataformat.xml.deser.FromXmlParser;
import javax.xml.stream.XMLStreamReader;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FromXmlParser_ESTest extends FromXmlParser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      FromXmlParser fromXmlParser0 = null;
      try {
        fromXmlParser0 = new FromXmlParser(iOContext0, 2, 3, (ObjectCodec) null, (XMLStreamReader) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.dataformat.xml.deser.XmlTokenStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      int int0 = FromXmlParser.Feature.collectDefaults();
      assertEquals(0, int0);
  }
}