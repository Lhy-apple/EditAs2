/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:15:19 GMT 2023
 */

package org.apache.commons.csv;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import java.io.CharArrayWriter;
import java.io.File;
import java.io.PipedWriter;
import java.io.StringWriter;
import java.nio.CharBuffer;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLInvalidAuthorizationSpecException;
import java.util.Stack;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.Quote;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileWriter;
import org.evosuite.runtime.mock.java.io.MockPrintWriter;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CSVPrinter_ESTest extends CSVPrinter_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVPrinter cSVPrinter0 = new CSVPrinter(charArrayWriter0, cSVFormat0);
      Appendable appendable0 = cSVPrinter0.getOut();
      assertSame(appendable0, charArrayWriter0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CharBuffer charBuffer0 = CharBuffer.wrap((CharSequence) "' in ");
      CSVPrinter cSVPrinter0 = cSVFormat0.print(charBuffer0);
      cSVPrinter0.close();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      CSVPrinter cSVPrinter0 = new CSVPrinter(charArrayWriter0, cSVFormat0);
      cSVPrinter0.close();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      StringWriter stringWriter0 = new StringWriter(10);
      StringBuffer stringBuffer0 = stringWriter0.getBuffer();
      CharBuffer charBuffer0 = CharBuffer.wrap((CharSequence) stringBuffer0);
      CSVPrinter cSVPrinter0 = cSVFormat0.print(charBuffer0);
      cSVPrinter0.flush();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MockFileWriter mockFileWriter0 = new MockFileWriter("\r\n");
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVPrinter cSVPrinter0 = cSVFormat0.print(mockFileWriter0);
      cSVPrinter0.flush();
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MockFileWriter mockFileWriter0 = new MockFileWriter("VAK+Ya");
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("Unexpected Quote value: ");
      CSVPrinter cSVPrinter0 = cSVFormat1.print(mockFileWriter0);
      Object[] objectArray0 = new Object[9];
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(9, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SQLInvalidAuthorizationSpecException sQLInvalidAuthorizationSpecException0 = new SQLInvalidAuthorizationSpecException();
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      CSVFormat cSVFormat0 = CSVFormat.newFormat('V');
      CSVPrinter cSVPrinter0 = new CSVPrinter(charArrayWriter0, cSVFormat0);
      cSVPrinter0.printRecords((Iterable<?>) sQLInvalidAuthorizationSpecException0);
      assertEquals(56, charArrayWriter0.size());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockFileWriter mockFileWriter0 = new MockFileWriter("ndex for header '%s' is %d but CVRecod only has %d values!");
      CSVFormat cSVFormat0 = CSVFormat.newFormat('c');
      Character character0 = new Character('0');
      CSVFormat cSVFormat1 = cSVFormat0.withEscape(character0);
      CSVPrinter cSVPrinter0 = cSVFormat1.print(mockFileWriter0);
      Object[] objectArray0 = new Object[1];
      objectArray0[0] = (Object) cSVFormat1;
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(1, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      Character character0 = new Character('6');
      CSVFormat cSVFormat1 = cSVFormat0.withEscape(character0);
      Quote quote0 = Quote.NONE;
      CSVFormat cSVFormat2 = cSVFormat1.withQuotePolicy(quote0);
      CSVFormat cSVFormat3 = cSVFormat2.withCommentStart((Character) null);
      CSVPrinter cSVPrinter0 = cSVFormat3.print(charArrayWriter0);
      Object[] objectArray0 = new Object[1];
      cSVPrinter0.printRecords(objectArray0);
      assertEquals("\r\n", charArrayWriter0.toString());
      assertEquals(2, charArrayWriter0.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockFileWriter mockFileWriter0 = new MockFileWriter("a!\"@P9nG)958[e");
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      Object[] objectArray0 = new Object[7];
      Quote quote0 = Quote.ALL;
      CSVFormat cSVFormat1 = cSVFormat0.withQuotePolicy(quote0);
      CSVPrinter cSVPrinter0 = cSVFormat1.print(mockFileWriter0);
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(7, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      Quote quote0 = Quote.NON_NUMERIC;
      CSVFormat cSVFormat1 = cSVFormat0.withQuotePolicy(quote0);
      CSVPrinter cSVPrinter0 = cSVFormat1.print(charArrayWriter0);
      Object[] objectArray0 = new Object[9];
      objectArray0[0] = (Object) (byte)20;
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(59, charArrayWriter0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MockFileWriter mockFileWriter0 = new MockFileWriter("a!\"@P9nG)958[e");
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      Object[] objectArray0 = new Object[12];
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("12!y|,&_sqn");
      CSVPrinter cSVPrinter0 = new CSVPrinter(mockFileWriter0, cSVFormat1);
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(12, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MockFileWriter mockFileWriter0 = new MockFileWriter(":cRzt?T||xbv?");
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      Object[] objectArray0 = new Object[14];
      CSVFormat cSVFormat1 = cSVFormat0.withNullString(":cRzt?T||xbv?");
      CSVPrinter cSVPrinter0 = cSVFormat1.print(mockFileWriter0);
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(14, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("_!' Mv;<3X(D:x");
      CSVPrinter cSVPrinter0 = new CSVPrinter(charArrayWriter0, cSVFormat1);
      Object[] objectArray0 = new Object[4];
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(72, charArrayWriter0.size());
      assertEquals("\"_!' Mv;<3X(D:x\"\r\n\"_!' Mv;<3X(D:x\"\r\n\"_!' Mv;<3X(D:x\"\r\n\"_!' Mv;<3X(D:x\"\r\n", charArrayWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      MockFileWriter mockFileWriter0 = new MockFileWriter("a!\"@P9nG)958[e");
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      Object[] objectArray0 = new Object[7];
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("|Y1EP'[2yF k]");
      CSVPrinter cSVPrinter0 = cSVFormat1.print(mockFileWriter0);
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(7, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      MockFileWriter mockFileWriter0 = new MockFileWriter("\r\n");
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("\r\n");
      Object[] objectArray0 = new Object[2];
      CSVPrinter cSVPrinter0 = cSVFormat1.print(mockFileWriter0);
      cSVPrinter0.printRecord(objectArray0);
      assertEquals(2, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      MockFileWriter mockFileWriter0 = new MockFileWriter("a!\"@P9nG)958[e");
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      Object[] objectArray0 = new Object[8];
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("a!\"@P9nG)958[e");
      CSVPrinter cSVPrinter0 = cSVFormat1.print(mockFileWriter0);
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(8, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      File file0 = MockFile.createTempFile("org.apache.commons.csv.CSVFormat", "format");
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter(file0);
      CSVPrinter cSVPrinter0 = cSVFormat0.print(mockPrintWriter0);
      cSVPrinter0.printComment("o`hB'-P`i+>:8");
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentStart('D');
      CSVPrinter cSVPrinter0 = cSVFormat1.print(charArrayWriter0);
      cSVPrinter0.print(cSVFormat1);
      cSVPrinter0.printComment("Delimiter=<,> QuoteChar=<\"> RecordSeparator=<\r\n> SkipHeaderRecord:false");
      assertEquals(122, charArrayWriter0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentStart('D');
      CSVPrinter cSVPrinter0 = cSVFormat1.print(charArrayWriter0);
      cSVPrinter0.printComment("Delimiter=<\t> Escape=<> RecordSeparator=<\n> SkipHeaderRecord:false");
      assertEquals(71, charArrayWriter0.size());
      assertEquals("D Delimiter=<\t> Escape=<> RecordSeparator=<\nD > SkipHeaderRecord:false\n", charArrayWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      CSVPrinter cSVPrinter0 = new CSVPrinter(charArrayWriter0, cSVFormat0);
      Stack<PipedWriter> stack0 = new Stack<PipedWriter>();
      PipedWriter pipedWriter0 = new PipedWriter();
      stack0.add(pipedWriter0);
      cSVPrinter0.printRecords((Iterable<?>) stack0);
      assertEquals(32, charArrayWriter0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      SQLInvalidAuthorizationSpecException sQLInvalidAuthorizationSpecException0 = new SQLInvalidAuthorizationSpecException();
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      CSVPrinter cSVPrinter0 = new CSVPrinter(charArrayWriter0, cSVFormat0);
      Object[] objectArray0 = new Object[12];
      objectArray0[0] = (Object) sQLInvalidAuthorizationSpecException0;
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(102, charArrayWriter0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      File file0 = MockFile.createTempFile("org.apache.commons.csv.CSVFormat", "format");
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter(file0);
      CSVPrinter cSVPrinter0 = cSVFormat0.print(mockPrintWriter0);
      ResultSetMetaData resultSetMetaData0 = mock(ResultSetMetaData.class, new ViolatedAssumptionAnswer());
      doReturn(8).when(resultSetMetaData0).getColumnCount();
      ResultSet resultSet0 = mock(ResultSet.class, new ViolatedAssumptionAnswer());
      doReturn(resultSetMetaData0).when(resultSet0).getMetaData();
      doReturn((String) null, (String) null, (String) null, (String) null, (String) null).when(resultSet0).getString(anyInt());
      doReturn(true, false).when(resultSet0).next();
      cSVPrinter0.printRecords(resultSet0);
  }
}