/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:56:53 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.PushbackInputStream;
import java.io.SequenceInputStream;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.util.Enumeration;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
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
        DataUtil.readToByteBuffer((InputStream) pushbackInputStream0);
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
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      DataUtil.load((InputStream) byteArrayInputStream0, (String) null, "");
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteBuffer byteBuffer0 = DataUtil.emptyByteBuffer();
      assertEquals(0, byteBuffer0.remaining());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      // Undeclared exception!
      try { 
        DataUtil.load((File) null, "8DR9$uhvkT2$u40K\"", "");
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
      PipedInputStream pipedInputStream0 = new PipedInputStream(536);
      Parser parser0 = Parser.htmlParser();
      try { 
        DataUtil.load((InputStream) pipedInputStream0, "q", "q", parser0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      File file0 = MockFile.createTempFile("--------------------------------", "!ya.lfc0xGrDfrJ");
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream(file0);
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(byteArrayInputStream0, (byte)96);
      DataUtil.crossStreams(bufferedInputStream0, mockFileOutputStream0);
      assertEquals(5L, file0.length());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = DataUtil.load((InputStream) null, "WxpaQcT5-z`x", "WxpaQcT5-z`x");
      assertEquals("WxpaQcT5-z`x", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      DataUtil.load((InputStream) byteArrayInputStream0, "UTF-8", "UTF-8");
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Enumeration<PushbackInputStream> enumeration0 = (Enumeration<PushbackInputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(sequenceInputStream0, 385);
      Parser parser0 = Parser.xmlParser();
      DataUtil.parseInputStream(pushbackInputStream0, (String) null, "en*+coding", parser0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, 5120);
      assertEquals(9, byteArray0.length);
      assertNotNull(byteArrayInputStream0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(9, byteArrayInputStream0.available());
      
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(byteArrayInputStream0, byteArrayInputStream0);
      assertEquals(9, byteArray0.length);
      assertNotNull(sequenceInputStream0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(9, byteArrayInputStream0.available());
      
      // Undeclared exception!
      try { 
        DataUtil.readToByteBuffer(sequenceInputStream0, (byte) (-99));
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
      File file0 = MockFile.createTempFile("!ya.lfc0xGrDfrJ", "!ya.lfc0xGrDfrJ");
      assertNotNull(file0);
      assertFalse(file0.isDirectory());
      assertTrue(file0.exists());
      assertEquals(1392409281320L, file0.lastModified());
      assertEquals(0L, file0.getTotalSpace());
      assertEquals(0L, file0.getFreeSpace());
      assertEquals("/tmp/!ya.lfc0xGrDfrJ0!ya.lfc0xGrDfrJ", file0.toString());
      assertEquals(0L, file0.length());
      assertTrue(file0.isAbsolute());
      assertTrue(file0.canExecute());
      assertEquals("!ya.lfc0xGrDfrJ0!ya.lfc0xGrDfrJ", file0.getName());
      assertTrue(file0.canWrite());
      assertTrue(file0.isFile());
      assertTrue(file0.canRead());
      assertFalse(file0.isHidden());
      assertEquals(0L, file0.getUsableSpace());
      assertEquals("/tmp", file0.getParent());
      
      ByteBuffer byteBuffer0 = DataUtil.readFileToByteBuffer(file0);
      assertNotNull(byteBuffer0);
      assertFalse(file0.isDirectory());
      assertTrue(file0.exists());
      assertEquals(1392409281320L, file0.lastModified());
      assertEquals(0L, file0.getTotalSpace());
      assertEquals(0L, file0.getFreeSpace());
      assertEquals("/tmp/!ya.lfc0xGrDfrJ0!ya.lfc0xGrDfrJ", file0.toString());
      assertEquals(0L, file0.length());
      assertTrue(file0.isAbsolute());
      assertTrue(file0.canExecute());
      assertEquals("!ya.lfc0xGrDfrJ0!ya.lfc0xGrDfrJ", file0.getName());
      assertTrue(file0.canWrite());
      assertTrue(file0.isFile());
      assertTrue(file0.canRead());
      assertFalse(file0.isHidden());
      assertEquals(0L, file0.getUsableSpace());
      assertEquals("/tmp", file0.getParent());
      assertEquals(0, byteBuffer0.capacity());
      assertEquals(0, byteBuffer0.remaining());
      assertFalse(byteBuffer0.isReadOnly());
      assertFalse(byteBuffer0.isDirect());
      assertTrue(byteBuffer0.hasArray());
      assertFalse(byteBuffer0.hasRemaining());
      assertEquals(0, byteBuffer0.limit());
      assertEquals(0, byteBuffer0.position());
      assertEquals(0, byteBuffer0.arrayOffset());
      assertEquals("java.nio.HeapByteBuffer[pos=0 lim=0 cap=0]", byteBuffer0.toString());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DataUtil.readFileToByteBuffer((File) null);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MockFile mockFile0 = new MockFile((String) null, "");
      assertNotNull(mockFile0);
      
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
      String string0 = DataUtil.getCharsetFromContentType("");
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
      assertNotNull(string0);
      assertEquals("--------------------------------", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, 1151);
      assertEquals(7, byteArray0.length);
      assertNotNull(byteArrayInputStream0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(7, byteArrayInputStream0.available());
      
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(byteArrayInputStream0, byteArrayInputStream0);
      assertEquals(7, byteArray0.length);
      assertNotNull(sequenceInputStream0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(7, byteArrayInputStream0.available());
      
      Parser parser0 = Parser.xmlParser();
      assertNotNull(parser0);
      assertFalse(parser0.isTrackErrors());
      
      try { 
        DataUtil.parseInputStream(sequenceInputStream0, "--------------------------------", "--------------------------------", parser0);
        fail("Expecting exception: UnsupportedEncodingException");
      
      } catch(UnsupportedEncodingException e) {
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      byteArray0[0] = (byte) (-27);
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      assertEquals(4, byteArray0.length);
      assertNotNull(byteArrayInputStream0);
      assertArrayEquals(new byte[] {(byte) (-27), (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(4, byteArrayInputStream0.available());
      
      // Undeclared exception!
      try { 
        DataUtil.load((InputStream) byteArrayInputStream0, "", "2[4@SOR+$Bx?i");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must set charset arg to character set of file to parse. Set to null to attempt to detect from HTML
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      byteArray0[1] = (byte) (-91);
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, 1151);
      assertEquals(7, byteArray0.length);
      assertNotNull(byteArrayInputStream0);
      assertArrayEquals(new byte[] {(byte)0, (byte) (-91), (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(7, byteArrayInputStream0.available());
      
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(byteArrayInputStream0, byteArrayInputStream0);
      assertEquals(7, byteArray0.length);
      assertNotNull(sequenceInputStream0);
      assertArrayEquals(new byte[] {(byte)0, (byte) (-91), (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(7, byteArrayInputStream0.available());
      
      Parser parser0 = Parser.xmlParser();
      assertNotNull(parser0);
      assertFalse(parser0.isTrackErrors());
      
      try { 
        DataUtil.parseInputStream(sequenceInputStream0, "--------------------------------", "--------------------------------", parser0);
        fail("Expecting exception: UnsupportedEncodingException");
      
      } catch(UnsupportedEncodingException e) {
      }
  }
}
