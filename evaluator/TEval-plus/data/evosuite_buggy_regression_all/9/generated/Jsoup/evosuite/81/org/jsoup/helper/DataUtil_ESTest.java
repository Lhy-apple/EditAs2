/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:15:15 GMT 2023
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
import java.io.SequenceInputStream;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
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
      MockFile mockFile0 = new MockFile("--------------------------------");
      mockFile0.createNewFile();
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(mockFile0);
      ByteBuffer byteBuffer0 = DataUtil.readToByteBuffer((InputStream) mockFileInputStream0);
      assertEquals(0, byteBuffer0.limit());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = DataUtil.load((InputStream) null, "org.jsoup.select.Evaluator$IsNthOfType", "UTF-8");
      assertEquals("UTF-8", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteBuffer byteBuffer0 = DataUtil.emptyByteBuffer();
      assertEquals(0, byteBuffer0.limit());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      File file0 = MockFile.createTempFile("UTF-8", "UTF-8");
      try { 
        DataUtil.load(file0, "09a@'U&", (String) null);
        fail("Expecting exception: UnsupportedEncodingException");
      
      } catch(UnsupportedEncodingException e) {
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(byteArrayInputStream0, byteArrayInputStream0);
      Parser parser0 = Parser.xmlParser();
      try { 
        DataUtil.load((InputStream) sequenceInputStream0, "E~.4B/}q#Un", "div", parser0);
        fail("Expecting exception: UnsupportedEncodingException");
      
      } catch(UnsupportedEncodingException e) {
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MockFile mockFile0 = new MockFile("--------------------------------", "--------------------------------");
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream(mockFile0, true);
      DataUtil.crossStreams(byteArrayInputStream0, mockFileOutputStream0);
      assertEquals(5L, mockFile0.length());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 0, 0);
      Parser parser0 = Parser.xmlParser();
      Document document0 = DataUtil.parseInputStream(byteArrayInputStream0, (String) null, "--------------------------------", parser0);
      assertEquals(0, document0.childNodeSize());
      assertEquals("--------------------------------", document0.location());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 0, 0);
      Parser parser0 = Parser.htmlParser();
      Document document0 = DataUtil.parseInputStream(byteArrayInputStream0, (String) null, "--------------------------------", parser0);
      assertEquals("--------------------------------", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      // Undeclared exception!
      try { 
        DataUtil.parseInputStream(byteArrayInputStream0, "UTF-16", (String) null, (Parser) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.helper.DataUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 0, 0);
      // Undeclared exception!
      try { 
        DataUtil.readToByteBuffer(byteArrayInputStream0, (-1));
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
      assertEquals(0, byteBuffer0.limit());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MockFile mockFile0 = new MockFile("Xpt+7Q$q+(a2|");
      try { 
        DataUtil.readFileToByteBuffer(mockFile0);
        fail("Expecting exception: FileNotFoundException");
      
      } catch(FileNotFoundException e) {
         //
         // File does not exist, and RandomAccessFile is not open in write mode
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockRandomAccessFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("/tmp/UTF-80UTF-8");
      FileSystemHandling.createFolder(evoSuiteFile0);
      File file0 = MockFile.createTempFile("UTF-8", "UTF-8");
      try { 
        DataUtil.readFileToByteBuffer(file0);
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
      String string0 = DataUtil.getCharsetFromContentType("--------------------------------");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType("charset=");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String string0 = DataUtil.mimeBoundary();
      assertEquals("--------------------------------", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      byteArray0[0] = (byte) (-25);
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      // Undeclared exception!
      try { 
        DataUtil.parseInputStream(byteArrayInputStream0, (String) null, "--------------------------------", (Parser) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.helper.DataUtil", e);
      }
  }
}