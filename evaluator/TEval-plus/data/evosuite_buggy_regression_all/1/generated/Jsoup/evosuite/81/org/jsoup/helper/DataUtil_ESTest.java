/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:52:34 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.jsoup.helper.DataUtil;
import org.jsoup.nodes.Document;
import org.jsoup.parser.Parser;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DataUtil_ESTest extends DataUtil_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      try { 
        DataUtil.readToByteBuffer((InputStream) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream closed
         //
         verifyException("java.io.BufferedInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, (byte) (-20));
      try { 
        DataUtil.load((InputStream) byteArrayInputStream0, ";", "}q@./p{@m*<c1");
        fail("Expecting exception: UnsupportedEncodingException");
      
      } catch(UnsupportedEncodingException e) {
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteBuffer byteBuffer0 = DataUtil.emptyByteBuffer();
      assertEquals(0, byteBuffer0.limit());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      // Undeclared exception!
      try { 
        DataUtil.load((File) null, "charset=", "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockFileInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Parser parser0 = Parser.xmlParser();
      DataUtil.load((InputStream) byteArrayInputStream0, (String) null, "", parser0);
      assertEquals(0, byteArrayInputStream0.available());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream((byte)0);
      DataUtil.crossStreams(byteArrayInputStream0, byteArrayOutputStream0);
      assertEquals("\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
      assertEquals(4, byteArrayOutputStream0.size());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Parser parser0 = Parser.htmlParser();
      Document document0 = DataUtil.load((InputStream) null, "S{", "S{", parser0);
      assertEquals("S{", document0.location());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Parser parser0 = Parser.xmlParser();
      byteArrayInputStream0.read(byteArray0);
      Document document0 = DataUtil.load((InputStream) byteArrayInputStream0, (String) null, "A8$", parser0);
      assertEquals("A8$", document0.baseUri());
      assertEquals(0, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Parser parser0 = Parser.xmlParser();
      Document document0 = DataUtil.load((InputStream) byteArrayInputStream0, "UTF-8", "UTF-8", parser0);
      assertFalse(document0.hasParent());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      // Undeclared exception!
      try { 
        DataUtil.readToByteBuffer(byteArrayInputStream0, (-2));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // maxSize must be 0 (unlimited) or larger
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      File file0 = MockFile.createTempFile("UTF-8", "UTF-8");
      ByteBuffer byteBuffer0 = DataUtil.readFileToByteBuffer(file0);
      assertTrue(byteBuffer0.hasArray());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      // Undeclared exception!
      try { 
        DataUtil.readFileToByteBuffer((File) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockRandomAccessFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MockFile mockFile0 = new MockFile("");
      try { 
        DataUtil.readFileToByteBuffer(mockFile0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.NativeMockedIO", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType("iu=8}24 8MFz13-8f");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      String string0 = DataUtil.mimeBoundary();
      assertEquals("--------------------------------", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      byteArray0[0] = (byte) (-1);
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Parser parser0 = Parser.htmlParser();
      try { 
        DataUtil.load((InputStream) byteArrayInputStream0, "=N", "=N", parser0);
        fail("Expecting exception: UnsupportedEncodingException");
      
      } catch(UnsupportedEncodingException e) {
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Parser parser0 = Parser.xmlParser();
      byte[] byteArray0 = new byte[13];
      byteArray0[1] = (byte) (-17);
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      try { 
        DataUtil.load((InputStream) byteArrayInputStream0, "--------------------------------", "--------------------------------", parser0);
        fail("Expecting exception: UnsupportedEncodingException");
      
      } catch(UnsupportedEncodingException e) {
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Parser parser0 = Parser.htmlParser();
      byte[] byteArray0 = new byte[8];
      byteArray0[0] = (byte) (-2);
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      try { 
        DataUtil.load((InputStream) byteArrayInputStream0, "maxSize must be 0 (unlimited) or larger", "maxSize must be 0 (unlimited) or larger", parser0);
        fail("Expecting exception: UnsupportedEncodingException");
      
      } catch(UnsupportedEncodingException e) {
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Parser parser0 = Parser.xmlParser();
      byte[] byteArray0 = new byte[13];
      byteArray0[0] = (byte) (-17);
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Document document0 = DataUtil.load((InputStream) byteArrayInputStream0, "UTF-8", "UTF-8", parser0);
      assertFalse(document0.updateMetaCharsetElement());
  }
}