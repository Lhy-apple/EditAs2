/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:33:48 GMT 2023
 */

package org.apache.commons.csv;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedReader;
import java.io.PushbackInputStream;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.util.Map;
import java.util.function.Consumer;
import javax.sql.rowset.RowSetMetaDataImpl;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.csv.QuoteMode;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.net.MockURL;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CSVParser_ESTest extends CSVParser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.POSTGRESQL_TEXT;
      CSVParser cSVParser0 = CSVParser.parse("eD;Re~K@FmDU?u5`Z", cSVFormat0);
      Consumer<Object> consumer0 = (Consumer<Object>) mock(Consumer.class, new ViolatedAssumptionAnswer());
      cSVParser0.forEach(consumer0);
      assertEquals(1L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.ORACLE;
      byte[] byteArray0 = new byte[8];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)2, 5);
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(byteArrayInputStream0);
      Charset charset0 = Charset.defaultCharset();
      CSVParser cSVParser0 = CSVParser.parse((InputStream) pushbackInputStream0, charset0, cSVFormat0);
      cSVParser0.getFirstEndOfLine();
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVParser cSVParser0 = CSVParser.parse("ML<kk%/j.p#$AD~", cSVFormat0);
      cSVParser0.getCurrentLineNumber();
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MockFile mockFile0 = new MockFile("`n@=m`&CWG7", "`n@=m`&CWG7");
      Charset charset0 = Charset.defaultCharset();
      CSVFormat cSVFormat0 = CSVFormat.POSTGRESQL_TEXT;
      try { 
        CSVParser.parse((File) mockFile0, charset0, cSVFormat0);
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
      CSVFormat cSVFormat0 = CSVFormat.ORACLE;
      URL uRL0 = MockURL.getFtpExample();
      // Undeclared exception!
      try { 
        CSVParser.parse(uRL0, (Charset) null, cSVFormat0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Parameter 'charset' must not be null!
         //
         verifyException("org.apache.commons.csv.Assertions", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.ORACLE;
      MockFile mockFile0 = new MockFile("=\"D|^|Pj*{");
      Path path0 = mockFile0.toPath();
      // Undeclared exception!
      try { 
        CSVParser.parse(path0, (Charset) null, cSVFormat0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.nio.file.Files", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVParser cSVParser0 = CSVParser.parse("^?", cSVFormat0);
      long long0 = cSVParser0.getRecordNumber();
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      CSVParser cSVParser0 = CSVParser.parse("rXMjmdKeE)f9g9,#", cSVFormat0);
      cSVParser0.nextRecord();
      assertEquals(1L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.ORACLE;
      CSVParser cSVParser0 = CSVParser.parse("\u0002\u0006\u0000\u0006\u0006", cSVFormat0);
      cSVParser0.nextRecord();
      assertEquals(1L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.ORACLE;
      CSVFormat cSVFormat1 = cSVFormat0.withTrailingDelimiter();
      byte[] byteArray0 = new byte[8];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)2, 5);
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(byteArrayInputStream0);
      Charset charset0 = Charset.defaultCharset();
      CSVParser cSVParser0 = CSVParser.parse((InputStream) pushbackInputStream0, charset0, cSVFormat1);
      cSVParser0.getRecords();
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      CSVFormat cSVFormat1 = cSVFormat0.withHeader((ResultSetMetaData) rowSetMetaDataImpl0);
      CSVParser cSVParser0 = CSVParser.parse("^?", cSVFormat1);
      Map<String, Integer> map0 = cSVParser0.getHeaderMap();
      assertEquals(1, map0.size());
      assertEquals(0L, cSVParser0.getRecordNumber());
      assertNotNull(map0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.POSTGRESQL_TEXT;
      CSVParser cSVParser0 = CSVParser.parse("N", cSVFormat0);
      cSVParser0.getHeaderMap();
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVParser cSVParser0 = CSVParser.parse("(*?30-A(", cSVFormat0);
      cSVParser0.getRecords();
      assertEquals(1L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      CSVFormat cSVFormat1 = cSVFormat0.withHeader((ResultSetMetaData) rowSetMetaDataImpl0);
      CSVFormat cSVFormat2 = cSVFormat1.withAutoFlush(true);
      CSVFormat cSVFormat3 = cSVFormat2.withIgnoreHeaderCase();
      CSVFormat cSVFormat4 = cSVFormat3.withTrim(false);
      PipedReader pipedReader0 = new PipedReader();
      try { 
        cSVFormat4.parse(pipedReader0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedReader", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.ORACLE;
      Class<QuoteMode> class0 = QuoteMode.class;
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(class0);
      PipedReader pipedReader0 = new PipedReader();
      CSVParser cSVParser0 = new CSVParser(pipedReader0, cSVFormat1);
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      CSVFormat cSVFormat1 = cSVFormat0.withHeader((ResultSetMetaData) rowSetMetaDataImpl0);
      CSVParser cSVParser0 = CSVParser.parse("", cSVFormat1);
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.ORACLE;
      Class<QuoteMode> class0 = QuoteMode.class;
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(class0);
      PipedReader pipedReader0 = new PipedReader();
      CSVFormat cSVFormat2 = cSVFormat1.withSkipHeaderRecord();
      CSVParser cSVParser0 = null;
      try {
        cSVParser0 = new CSVParser(pipedReader0, cSVFormat2);
        fail("Expecting exception: IOException");
      
      } catch(Throwable e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedReader", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      ResultSet resultSet0 = mock(ResultSet.class, new ViolatedAssumptionAnswer());
      doReturn(rowSetMetaDataImpl0).when(resultSet0).getMetaData();
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(resultSet0);
      CSVFormat cSVFormat2 = cSVFormat1.withQuote('d');
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)6, (byte)2);
      Charset charset0 = Charset.defaultCharset();
      CSVParser cSVParser0 = CSVParser.parse((InputStream) byteArrayInputStream0, charset0, cSVFormat2);
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.INFORMIX_UNLOAD;
      CSVParser cSVParser0 = CSVParser.parse("reader", cSVFormat0);
      cSVParser0.close();
      Consumer<CSVRecord> consumer0 = (Consumer<CSVRecord>) mock(Consumer.class, new ViolatedAssumptionAnswer());
      cSVParser0.forEach(consumer0);
      assertEquals(0L, cSVParser0.getRecordNumber());
  }
}