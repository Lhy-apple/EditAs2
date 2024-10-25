/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:30:31 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.time.ZoneId;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.System;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.evosuite.runtime.testdata.FileSystemHandling;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarArchiveOutputStream_ESTest extends TarArchiveOutputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0, false);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0);
      int int0 = tarArchiveOutputStream0.getCount();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0, false);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0);
      int int0 = tarArchiveOutputStream0.getRecordSize();
      assertEquals(512, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null, 508, "org.apache.commons.compress.archivers.zip.FallbackZipEncoding");
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, "YF1KXmDC5");
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      tarArchiveOutputStream0.setAddPaxHeadersForNonAsciiNames(true);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("", (byte)49);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(33188, tarArchiveEntry0.getMode());
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
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, 8754);
      tarArchiveOutputStream0.finish();
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
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("&MawdDhs", (byte)100);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      try { 
        tarArchiveOutputStream0.finish();
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
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0, false);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0);
      tarArchiveOutputStream0.close();
      tarArchiveOutputStream0.close();
      assertEquals(10240L, tarArchiveOutputStream0.getBytesWritten());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0, true);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0);
      tarArchiveOutputStream0.close();
      Map<String, String> map0 = ZoneId.SHORT_IDS;
      try { 
        tarArchiveOutputStream0.writePaxHeaders("reading from an output buffer", map0);
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
      MockFile mockFile0 = new MockFile("central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ", "central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // file name 'data/lhy/TEval-plus/central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length /central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ' is too long ( > 100 bytes)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      MockFile mockFile0 = new MockFile("central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ", "central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ");
      tarArchiveOutputStream0.setLongFileMode(1);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(1, TarArchiveOutputStream.BIGNUMBER_STAR);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      tarArchiveOutputStream0.setBigNumberMode(1);
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      tarArchiveOutputStream0.writePaxHeaders(".#}<&*\"RO+|x", hashMap0);
      assertEquals(1, TarArchiveOutputStream.LONGFILE_TRUNCATE);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      MockFile mockFile0 = new MockFile("central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ", "central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ");
      tarArchiveOutputStream0.setLongFileMode(3);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveOutputStream0.setAddPaxHeadersForNonAsciiNames(true);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(1000, TarArchiveEntry.MILLIS_PER_SECOND);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      MockFile mockFile0 = new MockFile("central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ", "central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ");
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length /central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ");
      FileSystemHandling.createFolder(evoSuiteFile0);
      tarArchiveOutputStream0.setLongFileMode(2);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertFalse(tarArchiveEntry0.isGNUSparse());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0, false);
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
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      MockFile mockFile0 = new MockFile("central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ", "central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ");
      tarArchiveOutputStream0.setLongFileMode(2);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveEntry0.setSize(1);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      try { 
        tarArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // entry 'data/lhy/TEval-plus/central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length /central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ' closed at '0' before the '1' bytes specified in the header were written
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      byte[] byteArray0 = new byte[1];
      try { 
        tarArchiveOutputStream0.write(byteArray0, (-817), 16877);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // request to write '16877' bytes exceeds size in header of '0' bytes for entry 'null'
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      Map<String, String> map0 = ZoneId.SHORT_IDS;
      tarArchiveOutputStream0.writePaxHeaders(" Masked: ", map0);
      assertEquals(0, tarArchiveOutputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      tarArchiveOutputStream0.writePaxHeaders(" \u0000", hashMap0);
      assertEquals(0, tarArchiveOutputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      MockFile mockFile0 = new MockFile("central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ", "central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ");
      TarArchiveEntry tarArchiveEntry0 = (TarArchiveEntry)tarArchiveOutputStream0.createArchiveEntry(mockFile0, "central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ");
      assertFalse(tarArchiveEntry0.isLink());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0, false);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0);
      tarArchiveOutputStream0.close();
      MockFile mockFile0 = new MockFile("BZU9\"25`<D#HqJ");
      try { 
        tarArchiveOutputStream0.createArchiveEntry(mockFile0, "' bytes specified in the header were written");
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream has already been finished
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      tarArchiveOutputStream0.setBigNumberMode(2);
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      System.setCurrentTimeMillis((-1270L));
      // Undeclared exception!
      tarArchiveOutputStream0.writePaxHeaders("U!&1X", hashMap0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      MockFile mockFile0 = new MockFile("central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ", "central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length ");
      tarArchiveOutputStream0.setLongFileMode(2);
      tarArchiveOutputStream0.setBigNumberMode(2);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveEntry0.setIds(2147418943, 10);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(33188, tarArchiveEntry0.getMode());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("Nk9k=~n:@-`t]K5A?g", (byte) (-107));
      tarArchiveEntry0.setUserId((byte) (-107));
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // user id '-107' is too big ( > 2097151 )
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("org.apache.commons.compress.archivers.zip.ZipEnodingHelper");
      tarArchiveEntry0.setIds(424935705, 424935705);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // group id '424935705' is too big ( > 2097151 )
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }
}
