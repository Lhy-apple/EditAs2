/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:53:07 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.IllegalCharsetNameException;
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
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Document document0 = DataUtil.load((InputStream) byteArrayInputStream0, "UTF-8", "UTF-8");
      assertEquals("#root", document0.tagName());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Parser parser0 = Parser.xmlParser();
      // Undeclared exception!
      try { 
        DataUtil.load((InputStream) null, "meta[http-equiv=content-type], meta[charset]", "meta[http-equiv=content-type], meta[charset]", parser0);
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
      // Undeclared exception!
      try { 
        DataUtil.load((File) null, "&Ha", "&Ha");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockFileInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      File file0 = MockFile.createTempFile("Z_WY$", "Z_WY$");
      // Undeclared exception!
      try { 
        DataUtil.load(file0, "Z_WY$", "Z_WY$");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // Z_WY$
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      File file0 = MockFile.createTempFile("DRA~RwwBWgl!q.z", "DRA~RwwBWgl!q.z");
      Document document0 = DataUtil.load(file0, (String) null, "DRA~RwwBWgl!q.z");
      assertEquals("DRA~RwwBWgl!q.z", document0.location());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)50, (byte)50);
      // Undeclared exception!
      try { 
        DataUtil.readToByteBuffer(byteArrayInputStream0, (byte) (-1));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // maxSize must be 0 (unlimited) or larger
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ByteBuffer byteBuffer0 = DataUtil.readToByteBuffer(byteArrayInputStream0, 4714);
      assertEquals(9, byteBuffer0.remaining());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      DataInputStream dataInputStream0 = new DataInputStream(byteArrayInputStream0);
      ByteBuffer byteBuffer0 = DataUtil.readToByteBuffer(dataInputStream0, 1);
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals("java.nio.HeapByteBuffer[pos=0 lim=1 cap=1]", byteBuffer0.toString());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType("BAWd*");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType("charset=");
      assertNull(string0);
  }
}