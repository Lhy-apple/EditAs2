/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:52:38 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.PushbackInputStream;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.jsoup.helper.DataUtil;
import org.jsoup.nodes.Document;
import org.jsoup.parser.Parser;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DataUtil_ESTest extends DataUtil_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream((InputStream) null);
      try { 
        DataUtil.load((InputStream) pushbackInputStream0, "version", "version");
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream closed
         //
         verifyException("java.io.PushbackInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ByteBuffer byteBuffer0 = DataUtil.emptyByteBuffer();
      assertEquals(0, byteBuffer0.remaining());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      Parser parser0 = Parser.xmlParser();
      try { 
        DataUtil.load((InputStream) pipedInputStream0, "[U6?#(Uo'oyyV/;|Ce", "K,p", parser0);
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
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(byteArrayInputStream0);
      MockPrintStream mockPrintStream0 = new MockPrintStream("html");
      DataUtil.crossStreams(pushbackInputStream0, mockPrintStream0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = DataUtil.parseInputStream((InputStream) null, "--------------------------------", "--------------------------------", (Parser) null);
      assertEquals("--------------------------------", document0.location());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      File file0 = MockFile.createTempFile("UTF-8", "UTF-8");
      Document document0 = DataUtil.load(file0, "UTF-8", "UTF-8");
      assertFalse(document0.updateMetaCharsetElement());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte) (-69), (byte) (-69));
      Parser parser0 = Parser.xmlParser();
      Document document0 = DataUtil.parseInputStream(byteArrayInputStream0, (String) null, "math", parser0);
      assertEquals(0, document0.childNodeSize());
      assertEquals("math", document0.location());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 1015, 0);
      // Undeclared exception!
      try { 
        DataUtil.readToByteBuffer(byteArrayInputStream0, (byte) (-126));
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
      String string0 = DataUtil.getCharsetFromContentType("charset=");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType("[U6?#(Uo'oyyV/;|Ce");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      String string0 = DataUtil.mimeBoundary();
      assertEquals("--------------------------------", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Parser parser0 = Parser.htmlParser();
      Document document0 = DataUtil.parseInputStream(byteArrayInputStream0, (String) null, "Hk8L", parser0);
      assertEquals("Hk8L", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      byteArray0[0] = (byte) (-1);
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Parser parser0 = Parser.htmlParser();
      Document document0 = DataUtil.parseInputStream(byteArrayInputStream0, (String) null, "Hk8L", parser0);
      assertEquals("Hk8L", document0.location());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[1] = (byte) (-1);
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(byteArrayInputStream0, (byte)17);
      Parser parser0 = Parser.htmlParser();
      try { 
        DataUtil.parseInputStream(bufferedInputStream0, "n01", "n01", parser0);
        fail("Expecting exception: UnsupportedEncodingException");
      
      } catch(UnsupportedEncodingException e) {
      }
  }
}
