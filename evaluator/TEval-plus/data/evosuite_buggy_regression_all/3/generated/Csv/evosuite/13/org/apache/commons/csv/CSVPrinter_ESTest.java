/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:31:55 GMT 2023
 */

package org.apache.commons.csv;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.FilterOutputStream;
import java.io.OutputStream;
import java.io.StringWriter;
import java.nio.CharBuffer;
import java.sql.ResultSet;
import java.sql.SQLInvalidAuthorizationSpecException;
import java.sql.SQLWarning;
import java.util.LinkedHashSet;
import javax.sql.rowset.RowSetMetaDataImpl;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.QuoteMode;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.evosuite.runtime.mock.java.io.MockPrintWriter;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CSVPrinter_ESTest extends CSVPrinter_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream((OutputStream) null, 10);
      MockPrintStream mockPrintStream0 = new MockPrintStream(bufferedOutputStream0, false);
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVPrinter cSVPrinter0 = new CSVPrinter(mockPrintStream0, cSVFormat0);
      Appendable appendable0 = cSVPrinter0.getOut();
      assertSame(appendable0, mockPrintStream0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      StringBuffer stringBuffer0 = stringWriter0.getBuffer();
      CharBuffer charBuffer0 = CharBuffer.wrap((CharSequence) stringBuffer0);
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      Object[] objectArray0 = new Object[2];
      objectArray0[0] = (Object) cSVFormat0;
      CSVFormat cSVFormat1 = cSVFormat0.withHeaderComments(objectArray0);
      CSVPrinter cSVPrinter0 = new CSVPrinter(charBuffer0, cSVFormat1);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      MockPrintStream mockPrintStream0 = new MockPrintStream("/I|H<p");
      String[] stringArray0 = new String[0];
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(stringArray0);
      CSVPrinter cSVPrinter0 = new CSVPrinter(mockPrintStream0, cSVFormat1);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter(204);
      CSVFormat cSVFormat0 = CSVFormat.newFormat('C');
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      ResultSet resultSet0 = mock(ResultSet.class, new ViolatedAssumptionAnswer());
      doReturn(rowSetMetaDataImpl0).when(resultSet0).getMetaData();
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(resultSet0);
      CSVFormat cSVFormat2 = cSVFormat1.withSkipHeaderRecord();
      CSVPrinter cSVPrinter0 = cSVFormat2.print(stringWriter0);
      assertNotNull(cSVPrinter0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CharBuffer charBuffer0 = CharBuffer.allocate(0);
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVPrinter cSVPrinter0 = new CSVPrinter(charBuffer0, cSVFormat0);
      cSVPrinter0.close();
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("<u`c,%v)#{%p:|z", true);
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter(mockFileOutputStream0, true);
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVPrinter cSVPrinter0 = new CSVPrinter(mockPrintWriter0, cSVFormat0);
      cSVPrinter0.close();
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      CharBuffer charBuffer0 = CharBuffer.allocate(18);
      CSVPrinter cSVPrinter0 = cSVFormat0.print(charBuffer0);
      cSVPrinter0.flush();
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("<u`c,%v)#{%p:|z", true);
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter(mockFileOutputStream0, true);
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVPrinter cSVPrinter0 = new CSVPrinter(mockPrintWriter0, cSVFormat0);
      cSVPrinter0.flush();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("gL+9P");
      MockPrintStream mockPrintStream0 = new MockPrintStream("KF *+=P:u2");
      Object[] objectArray0 = new Object[10];
      CSVPrinter cSVPrinter0 = new CSVPrinter(mockPrintStream0, cSVFormat1);
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(10, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      CSVFormat cSVFormat0 = CSVFormat.newFormat('C');
      CSVPrinter cSVPrinter0 = cSVFormat0.print(stringWriter0);
      SQLInvalidAuthorizationSpecException sQLInvalidAuthorizationSpecException0 = new SQLInvalidAuthorizationSpecException("Unexpected Token type: ", "\";dxLJ.L. dyyGzl Q");
      cSVPrinter0.printRecords((Iterable<?>) sQLInvalidAuthorizationSpecException0);
      assertEquals("Unexpected Token type: ", sQLInvalidAuthorizationSpecException0.getMessage());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withDelimiter('i');
      CharBuffer charBuffer0 = CharBuffer.allocate(56);
      CSVPrinter cSVPrinter0 = cSVFormat1.print(charBuffer0);
      cSVPrinter0.print(charBuffer0);
      assertEquals(21, charBuffer0.length());
      assertEquals(35, charBuffer0.position());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter(byteArrayOutputStream0, false);
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withIgnoreEmptyLines();
      CSVFormat cSVFormat2 = cSVFormat1.withEscape('2');
      QuoteMode quoteMode0 = QuoteMode.NONE;
      CSVFormat cSVFormat3 = cSVFormat2.withQuoteMode(quoteMode0);
      CSVPrinter cSVPrinter0 = cSVFormat3.print(mockPrintWriter0);
      Object[] objectArray0 = new Object[3];
      objectArray0[1] = (Object) cSVFormat2;
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(3, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      MockPrintStream mockPrintStream0 = new MockPrintStream("/I|H<p");
      Character character0 = new Character('0');
      CSVFormat cSVFormat1 = cSVFormat0.withEscape(character0);
      CSVPrinter cSVPrinter0 = new CSVPrinter(mockPrintStream0, cSVFormat1);
      Object[] objectArray0 = new Object[1];
      objectArray0[0] = (Object) cSVPrinter0;
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(1, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      StringBuffer stringBuffer0 = stringWriter0.getBuffer();
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      QuoteMode quoteMode0 = QuoteMode.ALL;
      CSVFormat cSVFormat1 = cSVFormat0.withQuoteMode(quoteMode0);
      CSVPrinter cSVPrinter0 = new CSVPrinter(stringBuffer0, cSVFormat1);
      Object[] objectArray0 = new Object[7];
      cSVPrinter0.printRecords(objectArray0);
      assertEquals("\"\"\r\n\"\"\r\n\"\"\r\n\"\"\r\n\"\"\r\n\"\"\r\n\"\"\r\n", stringBuffer0.toString());
      assertEquals("\"\"\r\n\"\"\r\n\"\"\r\n\"\"\r\n\"\"\r\n\"\"\r\n\"\"\r\n", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      QuoteMode quoteMode0 = QuoteMode.NON_NUMERIC;
      CSVFormat cSVFormat1 = cSVFormat0.withQuoteMode(quoteMode0);
      CSVPrinter cSVPrinter0 = new CSVPrinter(stringWriter0, cSVFormat1);
      Object[] objectArray0 = new Object[5];
      cSVPrinter0.printRecords(objectArray0);
      assertEquals("\"\"\r\n\"\"\r\n\"\"\r\n\"\"\r\n\"\"\r\n", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("<u`c,%v)#{%p:|z", true);
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter(mockFileOutputStream0, true);
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVPrinter cSVPrinter0 = new CSVPrinter(mockPrintWriter0, cSVFormat0);
      Object[] objectArray0 = new Object[6];
      cSVPrinter0.print(cSVFormat0);
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(6, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withQuote('0');
      SQLInvalidAuthorizationSpecException sQLInvalidAuthorizationSpecException0 = new SQLInvalidAuthorizationSpecException("x[gRPCuKq", "");
      SQLWarning sQLWarning0 = new SQLWarning("U 7%+ke)M)'b`#H:u#", sQLInvalidAuthorizationSpecException0);
      CSVPrinter cSVPrinter0 = cSVFormat1.print(stringWriter0);
      cSVPrinter0.printRecords((Iterable<?>) sQLWarning0);
      assertEquals(0, sQLWarning0.getErrorCode());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      StringBuffer stringBuffer0 = stringWriter0.getBuffer();
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(64);
      Object[] objectArray0 = new Object[8];
      objectArray0[1] = (Object) byteArrayOutputStream0;
      CSVFormat cSVFormat1 = cSVFormat0.withDelimiter('O');
      CSVPrinter cSVPrinter0 = cSVFormat1.print(stringBuffer0);
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(72, stringBuffer0.length());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      MockPrintStream mockPrintStream0 = new MockPrintStream("/I|H<p");
      DataOutputStream dataOutputStream0 = new DataOutputStream(mockPrintStream0);
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker('j');
      Object[] objectArray0 = new Object[8];
      objectArray0[0] = (Object) dataOutputStream0;
      CSVFormat cSVFormat2 = cSVFormat1.withHeaderComments(objectArray0);
      CSVPrinter cSVPrinter0 = new CSVPrinter(mockPrintStream0, cSVFormat2);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      MockPrintStream mockPrintStream0 = new MockPrintStream("/I|H<p");
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(mockPrintStream0);
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter(bufferedOutputStream0, false);
      CSVPrinter cSVPrinter0 = new CSVPrinter(mockPrintWriter0, cSVFormat0);
      LinkedHashSet<FilterOutputStream> linkedHashSet0 = new LinkedHashSet<FilterOutputStream>();
      linkedHashSet0.add(mockPrintStream0);
      cSVPrinter0.printRecords((Iterable<?>) linkedHashSet0);
      assertFalse(linkedHashSet0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter(byteArrayOutputStream0, false);
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      SQLInvalidAuthorizationSpecException sQLInvalidAuthorizationSpecException0 = new SQLInvalidAuthorizationSpecException("N{k", "N{k");
      CSVPrinter cSVPrinter0 = new CSVPrinter(mockPrintWriter0, cSVFormat0);
      Object[] objectArray0 = new Object[2];
      objectArray0[0] = (Object) sQLInvalidAuthorizationSpecException0;
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(2, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter(204);
      CSVFormat cSVFormat0 = CSVFormat.newFormat('C');
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      CSVPrinter cSVPrinter0 = new CSVPrinter(stringWriter0, cSVFormat0);
      ResultSet resultSet0 = mock(ResultSet.class, new ViolatedAssumptionAnswer());
      doReturn(rowSetMetaDataImpl0).when(resultSet0).getMetaData();
      doReturn(false).when(resultSet0).next();
      cSVPrinter0.printRecords(resultSet0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter(204);
      CSVFormat cSVFormat0 = CSVFormat.newFormat('C');
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      CSVPrinter cSVPrinter0 = new CSVPrinter(stringWriter0, cSVFormat0);
      ResultSet resultSet0 = mock(ResultSet.class, new ViolatedAssumptionAnswer());
      doReturn(rowSetMetaDataImpl0).when(resultSet0).getMetaData();
      doReturn(true, false, false, false, false).when(resultSet0).next();
      // Undeclared exception!
      cSVPrinter0.printRecords(resultSet0);
  }
}