/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:50:41 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.PushbackInputStream;
import java.nio.ByteBuffer;
import java.nio.charset.UnsupportedCharsetException;
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
  public void test00()  throws Throwable  {
      File file0 = MockFile.createTempFile("replacementChar", "replacementChar");
      Document document0 = DataUtil.load(file0, (String) null, "replacementChar");
      assertEquals("replacementChar", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      // Undeclared exception!
      try { 
        DataUtil.load((InputStream) null, "", (String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.helper.DataUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(pipedInputStream0);
      Parser parser0 = Parser.htmlParser();
      try { 
        DataUtil.load((InputStream) pushbackInputStream0, "http-equiv", "*-e[z#FXv1l;ziEp", parser0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MockFile mockFile0 = new MockFile("UTF-8");
      try { 
        DataUtil.load((File) mockFile0, "UTF-8", "UTF-8");
        fail("Expecting exception: FileNotFoundException");
      
      } catch(FileNotFoundException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockFileInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      File file0 = MockFile.createTempFile("replacementChar", "replacementChar");
      // Undeclared exception!
      try { 
        DataUtil.load(file0, "replacementChar", "replacementChar");
        fail("Expecting exception: UnsupportedCharsetException");
      
      } catch(UnsupportedCharsetException e) {
         //
         // replacementChar
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      File file0 = MockFile.createTempFile("UTF-8", "UTF-8");
      Document document0 = DataUtil.load(file0, "UTF-8", "UTF-8");
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("/tmp/UTF-80UTF-8");
      FileSystemHandling.appendLineToFile(evoSuiteFile0, "UTF-8");
      File file0 = MockFile.createTempFile("UTF-8", "UTF-8");
      Document document0 = DataUtil.load(file0, "UTF-8", "UTF-8");
      assertEquals("UTF-8", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      // Undeclared exception!
      try { 
        DataUtil.readToByteBuffer(pipedInputStream0, (-493));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // maxSize must be 0 (unlimited) or larger
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ByteBuffer byteBuffer0 = DataUtil.readToByteBuffer(byteArrayInputStream0, (byte)77);
      assertEquals(6, byteBuffer0.capacity());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ByteBuffer byteBuffer0 = DataUtil.readToByteBuffer(byteArrayInputStream0, 1);
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals("java.nio.HeapByteBuffer[pos=0 lim=1 cap=1]", byteBuffer0.toString());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType("");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType((String) null);
      assertNull(string0);
  }
}