/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:48:33 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.nio.ByteBuffer;
import java.nio.charset.IllegalCharsetNameException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.jsoup.helper.DataUtil;
import org.jsoup.nodes.Document;
import org.jsoup.parser.Parser;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DataUtil_ESTest extends DataUtil_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      File file0 = MockFile.createTempFile(":X5?A4,~mc< EFMs", ":X5?A4,~mc< EFMs");
      // Undeclared exception!
      try { 
        DataUtil.load(file0, ":X5?A4,~mc< EFMs", ":X5?A4,~mc< EFMs");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // :X5?A4,~mc< EFMs
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      try { 
        DataUtil.load((InputStream) pipedInputStream0, "", "");
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MockFile mockFile0 = new MockFile("<$]k{b8");
      File file0 = MockFile.createTempFile("<$]k{b8", "<$]k{b8", (File) mockFile0);
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(file0);
      Parser parser0 = Parser.htmlParser();
      // Undeclared exception!
      try { 
        DataUtil.load((InputStream) mockFileInputStream0, "<$]k{b8", "^>mOJq_|LtHWI/", parser0);
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // <$]k{b8
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      File file0 = MockFile.createTempFile("UTF-8", "UTF-8");
      Document document0 = DataUtil.load(file0, "UTF-8", "UTF-8");
      assertFalse(document0.isBlock());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MockFile mockFile0 = new MockFile("~J~pxFbrP");
      try { 
        DataUtil.load((File) mockFile0, "~J~pxFbrP", "~J~pxFbrP");
        fail("Expecting exception: FileNotFoundException");
      
      } catch(FileNotFoundException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockFileInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      File file0 = MockFile.createTempFile("7+pA4N~mc< QEDMs", (String) null);
      Document document0 = DataUtil.load(file0, (String) null, "7+pA4N~mc< QEDMs");
      assertEquals("7+pA4N~mc< QEDMs", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ByteBuffer byteBuffer0 = ByteBuffer.allocate(2878);
      // Undeclared exception!
      try { 
        DataUtil.parseByteData(byteBuffer0, "UTF-8", "org.jsoup.helper.DataUtil", (Parser) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.helper.DataUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream((InputStream) null);
      // Undeclared exception!
      try { 
        DataUtil.readToByteBuffer(bufferedInputStream0, (-1));
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
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ByteBuffer byteBuffer0 = DataUtil.readToByteBuffer(byteArrayInputStream0, (byte)7);
      assertEquals(1, byteBuffer0.capacity());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ByteBuffer byteBuffer0 = DataUtil.readToByteBuffer(byteArrayInputStream0, (byte)0);
      assertEquals(1, byteBuffer0.limit());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ByteBuffer byteBuffer0 = DataUtil.readToByteBuffer(byteArrayInputStream0, 1);
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(1, byteBuffer0.limit());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType("~J~pxFbrP");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType((String) null);
      assertNull(string0);
  }
}