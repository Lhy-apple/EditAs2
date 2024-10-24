/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:34:04 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.nio.ByteBuffer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.evosuite.runtime.testdata.FileSystemHandling;
import org.jsoup.helper.DataUtil;
import org.jsoup.nodes.Document;
import org.jsoup.parser.Parser;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DataUtil_ESTest extends DataUtil_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      // Undeclared exception!
      try { 
        DataUtil.load((InputStream) null, "OnRF6]rC_{Bl_", "OnRF6]rC_{Bl_");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.helper.DataUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Parser parser0 = Parser.xmlParser();
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      try { 
        DataUtil.load((InputStream) pipedInputStream0, ".s/3+SX(p5.7g9Y2iA", "", parser0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      // Undeclared exception!
      try { 
        DataUtil.load((File) null, "6eOrv", "6eOrv");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockFileInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      File file0 = MockFile.createTempFile("lRfwM@fj[4Ei", "lRfwM@fj[4Ei");
      // Undeclared exception!
      try { 
        DataUtil.load(file0, "UTF-8", "lRfwM@fj[4Ei");
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ByteBuffer byteBuffer0 = ByteBuffer.wrap(byteArray0);
      Parser parser0 = Parser.xmlParser();
      Document document0 = DataUtil.parseByteData(byteBuffer0, "UTF-8", "UTF-8", parser0);
      assertEquals(0, document0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("/tmp/-.30-.3");
      FileSystemHandling.appendLineToFile(evoSuiteFile0, "-.3");
      File file0 = MockFile.createTempFile("-.3", "-.3");
      Document document0 = DataUtil.load(file0, (String) null, "-.3");
      assertEquals("-.3", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType("-.3");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType((String) null);
      assertNull(string0);
  }
}
