/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:27:18 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PushbackInputStream;
import java.io.SequenceInputStream;
import java.net.URI;
import java.nio.channels.FileChannel;
import java.nio.channels.SeekableByteChannel;
import java.util.Enumeration;
import java.util.zip.ZipException;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.jar.JarArchiveEntry;
import org.apache.commons.compress.archivers.zip.JarMarker;
import org.apache.commons.compress.archivers.zip.Zip64ExtendedInformationExtraField;
import org.apache.commons.compress.archivers.zip.Zip64Mode;
import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream;
import org.apache.commons.compress.archivers.zip.ZipExtraField;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.evosuite.runtime.mock.java.net.MockURI;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ZipArchiveOutputStream_ESTest extends ZipArchiveOutputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      zipArchiveOutputStream0.setMethod(34);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      zipArchiveOutputStream0.setFallbackToUTF8(false);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      zipArchiveOutputStream0.deflate();
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = null;
      try {
        zipArchiveOutputStream0 = new ZipArchiveOutputStream((File) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      String string0 = zipArchiveOutputStream0.getEncoding();
      assertEquals("UTF8", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream(" Q2oe~Fh=]3", false);
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFileOutputStream0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry(" Q2oe~Fh=]3");
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(" Q2oe~Fh=]3");
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(mockFileInputStream0);
      DataInputStream dataInputStream0 = new DataInputStream(bufferedInputStream0);
      zipArchiveOutputStream0.addRawArchiveEntry(jarArchiveEntry0, dataInputStream0);
      try { 
        zipArchiveOutputStream0.close();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Error in writing to file
         //
         verifyException("org.evosuite.runtime.mock.java.io.NativeMockedIO", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.writeOut((byte[]) null, (-1), 1000);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.StreamCompressor$OutputStreamCompressor", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      zipArchiveOutputStream0.setComment("vl %ihbn5'lVoV");
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.writeLocalFileHeader(zipArchiveEntry0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream$CurrentEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ZipArchiveOutputStream.UnicodeExtraFieldPolicy zipArchiveOutputStream_UnicodeExtraFieldPolicy0 = ZipArchiveOutputStream.UnicodeExtraFieldPolicy.NOT_ENCODEABLE;
      String string0 = zipArchiveOutputStream_UnicodeExtraFieldPolicy0.toString();
      assertEquals("not encodeable", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      boolean boolean0 = zipArchiveOutputStream0.isSeekable();
      assertFalse(boolean0);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("-5ehv1O9Xdd", false);
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      boolean boolean0 = zipArchiveOutputStream0.isSeekable();
      assertTrue(boolean0);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'", true);
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      zipArchiveOutputStream0.setUseLanguageEncodingFlag(false);
      zipArchiveOutputStream0.setEncoding((String) null);
      assertEquals((-1), ZipArchiveOutputStream.DEFAULT_COMPRESSION);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      zipArchiveOutputStream0.setEncoding("Z");
      assertEquals("Z", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((SeekableByteChannel) null);
      zipArchiveOutputStream0.setEncoding((String) null);
      assertEquals(0, ZipArchiveOutputStream.STORED);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'");
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      zipArchiveOutputStream0.setUseLanguageEncodingFlag(true);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.close();
      try { 
        zipArchiveOutputStream0.finish();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // This archive has already been finished
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'");
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      try { 
        zipArchiveOutputStream0.close();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // This archive contains unclosed entries.
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream(" Q2oe~Fh=]3", false);
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFileOutputStream0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry(" Q2oe~Fh=]3");
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(" Q2oe~Fh=]3");
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(mockFileInputStream0);
      DataInputStream dataInputStream0 = new DataInputStream(bufferedInputStream0);
      zipArchiveOutputStream0.addRawArchiveEntry(jarArchiveEntry0, dataInputStream0);
      zipArchiveOutputStream0.addRawArchiveEntry(jarArchiveEntry0, bufferedInputStream0);
      assertEquals(2L, jarArchiveEntry0.getCompressedSize());
      assertEquals(6646, zipArchiveOutputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      try { 
        zipArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No current entry to close
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("MMm!.", true);
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFileOutputStream0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setCrc(8);
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream("MMm!.");
      zipArchiveOutputStream0.addRawArchiveEntry(zipArchiveEntry0, mockFileInputStream0);
      assertEquals(2L, zipArchiveEntry0.getCompressedSize());
      assertEquals(3081, zipArchiveOutputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'", false);
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(sequenceInputStream0, 3);
      zipArchiveEntry0.setCompressedSize((-1));
      zipArchiveOutputStream0.addRawArchiveEntry(zipArchiveEntry0, pushbackInputStream0);
      assertEquals(8, zipArchiveEntry0.getMethod());
      assertEquals(114L, fileChannel0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'", false);
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("29X&MWNR$+'");
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      jarArchiveEntry0.setMethod(0);
      zipArchiveOutputStream0.addRawArchiveEntry(jarArchiveEntry0, sequenceInputStream0);
      assertEquals(0L, jarArchiveEntry0.getSize());
      assertEquals(122L, fileChannel0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("archive's size exceeds the limit of 4GByte.");
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      zipArchiveOutputStream0.writeCentralFileHeader(zipArchiveEntry0);
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      assertEquals(52L, zipArchiveEntry0.getCompressedSize());
      assertEquals(132L, fileChannel0.position());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveOutputStream0.finished = true;
      try { 
        zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream has already been finished
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((SeekableByteChannel) null);
      zipArchiveOutputStream0.setLevel(8);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.StreamCompressor$SeekableByteChannelCompressor", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'", false);
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("29X&MWNR$+'");
      jarArchiveEntry0.setMethod(0);
      ZipArchiveOutputStream zipArchiveOutputStream1 = new ZipArchiveOutputStream(zipArchiveOutputStream0);
      try { 
        zipArchiveOutputStream1.putArchiveEntry(jarArchiveEntry0);
        fail("Expecting exception: ZipException");
      
      } catch(ZipException e) {
         //
         // uncompressed size is required for STORED method when not writing to a file
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'", false);
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("29X&MWNR$+'");
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      jarArchiveEntry0.setMethod(0);
      ZipArchiveOutputStream zipArchiveOutputStream1 = new ZipArchiveOutputStream(zipArchiveOutputStream0);
      zipArchiveOutputStream1.putArchiveEntry(jarArchiveEntry0);
      assertEquals(0L, jarArchiveEntry0.getCompressedSize());
      assertEquals(145L, fileChannel0.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'");
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      Zip64Mode zip64Mode0 = Zip64Mode.Always;
      zipArchiveOutputStream0.setUseZip64(zip64Mode0);
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      zipArchiveOutputStream0.writeCentralFileHeader(zipArchiveEntry0);
      assertEquals(8, zipArchiveEntry0.getMethod());
      assertEquals(124L, fileChannel0.size());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'");
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      Zip64Mode zip64Mode0 = Zip64Mode.Never;
      zipArchiveOutputStream0.setUseZip64(zip64Mode0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      assertEquals(30L, fileChannel0.size());
      assertEquals(8, zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.setLevel((-418));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid compression level: -418
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.setLevel(1272);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid compression level: 1272
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      zipArchiveOutputStream0.setLevel((-1));
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      boolean boolean0 = zipArchiveOutputStream0.canWriteEntryData((ArchiveEntry) null);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'");
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      boolean boolean0 = zipArchiveOutputStream0.canWriteEntryData(zipArchiveEntry0);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setMethod(6);
      boolean boolean0 = zipArchiveOutputStream0.canWriteEntryData(zipArchiveEntry0);
      assertFalse(boolean0);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'");
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      boolean boolean0 = zipArchiveOutputStream0.canWriteEntryData(zipArchiveEntry0);
      assertEquals(50L, fileChannel0.position());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'", false);
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("29X&MWNR$+'");
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      zipArchiveOutputStream0.addRawArchiveEntry(jarArchiveEntry0, sequenceInputStream0);
      ZipArchiveOutputStream zipArchiveOutputStream1 = new ZipArchiveOutputStream(zipArchiveOutputStream0);
      // Undeclared exception!
      try { 
        zipArchiveOutputStream1.putArchiveEntry(jarArchiveEntry0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No current entry
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.finish();
      zipArchiveOutputStream0.close();
      assertEquals(22, byteArrayOutputStream0.size());
      assertEquals("\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((SeekableByteChannel) null);
      zipArchiveOutputStream0.flush();
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFileOutputStream0);
      zipArchiveOutputStream0.flush();
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'", false);
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setAlignment(8);
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream((InputStream) null, 742);
      try { 
        zipArchiveOutputStream0.addRawArchiveEntry(zipArchiveEntry0, bufferedInputStream0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream closed
         //
         verifyException("java.io.BufferedInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("archive's size exceehs the limit of 4GByte.");
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("archive's size exceehs the limit of 4GByte.");
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      jarArchiveEntry0.setMethod(0);
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      assertEquals(0, jarArchiveEntry0.getMethod());
      assertEquals(166L, fileChannel0.position());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      ZipArchiveOutputStream.UnicodeExtraFieldPolicy zipArchiveOutputStream_UnicodeExtraFieldPolicy0 = ZipArchiveOutputStream.UnicodeExtraFieldPolicy.NOT_ENCODEABLE;
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      zipArchiveOutputStream0.setCreateUnicodeExtraFields(zipArchiveOutputStream_UnicodeExtraFieldPolicy0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'");
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      ZipArchiveOutputStream.UnicodeExtraFieldPolicy zipArchiveOutputStream_UnicodeExtraFieldPolicy0 = ZipArchiveOutputStream.UnicodeExtraFieldPolicy.ALWAYS;
      zipArchiveOutputStream0.setCreateUnicodeExtraFields(zipArchiveOutputStream_UnicodeExtraFieldPolicy0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("29X&MWNR$+'");
      jarArchiveEntry0.setComment("29X&MWNR$+'");
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      assertEquals(8, jarArchiveEntry0.getMethod());
      assertEquals(101L, fileChannel0.position());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      zipArchiveOutputStream0.setCreateUnicodeExtraFields((ZipArchiveOutputStream.UnicodeExtraFieldPolicy) null);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setComment("");
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      ZipArchiveOutputStream.UnicodeExtraFieldPolicy zipArchiveOutputStream_UnicodeExtraFieldPolicy0 = ZipArchiveOutputStream.UnicodeExtraFieldPolicy.NOT_ENCODEABLE;
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      zipArchiveOutputStream0.setCreateUnicodeExtraFields(zipArchiveOutputStream_UnicodeExtraFieldPolicy0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setComment("Z");
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'", true);
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFileOutputStream0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("29X&MWNR$+'");
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      ZipExtraField[] zipExtraFieldArray0 = new ZipExtraField[2];
      JarMarker jarMarker0 = JarMarker.getInstance();
      zipExtraFieldArray0[0] = (ZipExtraField) jarMarker0;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField();
      zipExtraFieldArray0[1] = (ZipExtraField) zip64ExtendedInformationExtraField0;
      jarArchiveEntry0.setExtraFields(zipExtraFieldArray0);
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      assertEquals(0L, jarArchiveEntry0.getCrc());
      assertEquals(2L, jarArchiveEntry0.getCompressedSize());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'");
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("29X&MWNR$+'");
      jarArchiveEntry0.setComment("29X&MWNR$+'");
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      zipArchiveOutputStream0.writeCentralFileHeader(jarArchiveEntry0);
      assertEquals(133L, fileChannel0.size());
      assertEquals(133L, fileChannel0.position());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("~mxcByjY4FqV", false);
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      Zip64Mode zip64Mode0 = Zip64Mode.Never;
      zipArchiveOutputStream0.setUseZip64(zip64Mode0);
      zipArchiveOutputStream0.finish();
      assertEquals(22L, fileChannel0.size());
      assertEquals(22L, fileChannel0.position());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+J'");
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      zipArchiveOutputStream0.writeZip64CentralDirectory();
      assertEquals(126L, fileChannel0.size());
      assertEquals(126L, fileChannel0.position());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("29X&MWNR$+'", false);
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("29X&MWNR$+'");
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      zipArchiveOutputStream0.setUseLanguageEncodingFlag(false);
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      zipArchiveOutputStream0.addRawArchiveEntry(jarArchiveEntry0, sequenceInputStream0);
      assertEquals(2L, jarArchiveEntry0.getCompressedSize());
      assertEquals(124L, fileChannel0.size());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      MockFile mockFile0 = new MockFile(" GID=");
      ArchiveEntry archiveEntry0 = zipArchiveOutputStream0.createArchiveEntry(mockFile0, " GID=");
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
      assertEquals(" GID=", archiveEntry0.getName());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      URI uRI0 = MockURI.aFileURI;
      MockFile mockFile0 = new MockFile(uRI0);
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockPrintStream0);
      zipArchiveOutputStream0.close();
      try { 
        zipArchiveOutputStream0.createArchiveEntry(mockFile0, "%V+0kLkNd5|7");
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream has already been finished
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("archive's size exceeds the limit of 4GByte.");
      FileChannel fileChannel0 = mockFileOutputStream0.getChannel();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(fileChannel0);
      zipArchiveOutputStream0.close();
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }
}
