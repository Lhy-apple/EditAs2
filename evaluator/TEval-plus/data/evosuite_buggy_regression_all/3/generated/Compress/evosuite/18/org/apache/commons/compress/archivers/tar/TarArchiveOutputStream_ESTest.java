/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:28:55 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.time.ZoneId;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.System;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.evosuite.runtime.testdata.FileSystemHandling;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarArchiveOutputStream_ESTest extends TarArchiveOutputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(1264);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, 1264, 1264);
      int int0 = tarArchiveOutputStream0.getCount();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(1264);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, 1264, 1264);
      int int0 = tarArchiveOutputStream0.getRecordSize();
      assertEquals(1264, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, 2, (String) null);
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null, "ASCII");
      assertEquals(0, TarArchiveOutputStream.BIGNUMBER_ERROR);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      tarArchiveOutputStream0.setAddPaxHeadersForNonAsciiNames(true);
      tarArchiveOutputStream0.setLongFileMode(3);
      File file0 = MockFile.createTempFile("org.apachexcommons.compress.archivers.dump.DumpArchiveConstants$COMPRESSION_TYPE", "1|E%\"%>J,g_7I>H");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(file0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals("", tarArchiveEntry0.getLinkName());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.flush();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.io.FilterOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("QNU;fHm;,y'v<'|,?", true);
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFileOutputStream0, true);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0);
      tarArchiveOutputStream0.close();
      try { 
        tarArchiveOutputStream0.finish();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // This archive has already been finished
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null, 8192);
      File file0 = MockFile.createTempFile("bZ,)QUxKElz\"/u?gf", "bZ,)QUxKElz\"/u?gf");
      ArchiveEntry archiveEntry0 = tarArchiveOutputStream0.createArchiveEntry(file0, "bZ,)QUxKElz\"/u?gf");
      tarArchiveOutputStream0.putArchiveEntry(archiveEntry0);
      try { 
        tarArchiveOutputStream0.close();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // This archives contains unclosed entries.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("mbJ", true);
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(mockFileOutputStream0);
      MockPrintStream mockPrintStream0 = new MockPrintStream(bufferedOutputStream0, true);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0);
      tarArchiveOutputStream0.close();
      tarArchiveOutputStream0.close();
      assertEquals(10240, tarArchiveOutputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("mbJ");
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(mockFileOutputStream0);
      MockPrintStream mockPrintStream0 = new MockPrintStream(bufferedOutputStream0, true);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0);
      tarArchiveOutputStream0.close();
      Map<String, String> map0 = ZoneId.SHORT_IDS;
      try { 
        tarArchiveOutputStream0.writePaxHeaders("mbJ", map0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream has already been finished
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      MockFile mockFile0 = new MockFile("org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader", "org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // file name 'data/lhy/TEval-plus/org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader/org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader' is too long ( > 100 bytes)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      tarArchiveOutputStream0.setLongFileMode(3);
      File file0 = MockFile.createTempFile("org.apachexcommons.compress.archivers.dump.DumpArchiveConstants$COMPRESSION_TYPE", "1|E%\"%>J,g_7I>H");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(file0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals("lhy", tarArchiveEntry0.getUserName());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      MockFile mockFile0 = new MockFile("org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader", "org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader");
      tarArchiveOutputStream0.setLongFileMode(1);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(0L, tarArchiveEntry0.getRealSize());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      Map<String, String> map0 = ZoneId.SHORT_IDS;
      tarArchiveOutputStream0.setBigNumberMode(1);
      try { 
        tarArchiveOutputStream0.writePaxHeaders("", map0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // request to write '637' bytes exceeds size in header of '0' bytes for entry './PaxHeaders.X/'
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      MockFile mockFile0 = new MockFile("org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader", "org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader");
      tarArchiveOutputStream0.setAddPaxHeadersForNonAsciiNames(true);
      TarArchiveEntry tarArchiveEntry0 = (TarArchiveEntry)tarArchiveOutputStream0.createArchiveEntry(mockFile0, "org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader");
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(1000, TarArchiveEntry.MILLIS_PER_SECOND);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("mbJ", true);
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(mockFileOutputStream0);
      MockPrintStream mockPrintStream0 = new MockPrintStream(bufferedOutputStream0, true);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0);
      tarArchiveOutputStream0.close();
      try { 
        tarArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream has already been finished
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      try { 
        tarArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No current entry to close
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader/org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      MockFile mockFile0 = new MockFile("org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader", "org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader");
      byte[] byteArray0 = new byte[1];
      FileSystemHandling.appendDataToFile(evoSuiteFile0, byteArray0);
      tarArchiveOutputStream0.setLongFileMode(2);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      try { 
        tarArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // entry 'data/lhy/TEval-plus/org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader/org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader' closed at '0' before the '1' bytes specified in the header were written
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      Map<String, String> map0 = ZoneId.SHORT_IDS;
      tarArchiveOutputStream0.writePaxHeaders("0\u0000", map0);
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      tarArchiveOutputStream0.finish();
      MockFile mockFile0 = new MockFile("");
      try { 
        tarArchiveOutputStream0.createArchiveEntry(mockFile0, "");
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream has already been finished
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      HashMap<String, String> hashMap0 = new HashMap<String, String>(2);
      tarArchiveOutputStream0.setBigNumberMode(2);
      System.setCurrentTimeMillis((-1925L));
      // Undeclared exception!
      tarArchiveOutputStream0.writePaxHeaders("", hashMap0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      tarArchiveOutputStream0.setBigNumberMode(2);
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      System.setCurrentTimeMillis(9151314442816847872L);
      // Undeclared exception!
      tarArchiveOutputStream0.writePaxHeaders("STICKY", hashMap0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      Map<String, String> map0 = ZoneId.SHORT_IDS;
      System.setCurrentTimeMillis((-3443L));
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.writePaxHeaders("org.apache.commons.compress.archivers.dump.DumpArchiveConstants$SEGMENT_TYPE", map0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // last modification time '-3' is too big ( > 8589934591 )
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      MockFile mockFile0 = new MockFile("org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader", "org.apache.commons.K:mpress.archivers.dump.DumpArchiveEnty$TapeSegmentHader");
      tarArchiveOutputStream0.setLongFileMode(2);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      MockDate mockDate0 = new MockDate((-359), 2147356412, 2, 0, 1);
      tarArchiveEntry0.setModTime((Date) mockDate0);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // last modification time '5646988397145660' is too big ( > 8589934591 )
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }
}