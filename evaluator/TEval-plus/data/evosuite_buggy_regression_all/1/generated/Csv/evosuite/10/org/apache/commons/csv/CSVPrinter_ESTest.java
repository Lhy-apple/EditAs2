/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:28:06 GMT 2023
 */

package org.apache.commons.csv;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import java.io.ByteArrayOutputStream;
import java.io.CharArrayWriter;
import java.io.StringWriter;
import java.nio.CharBuffer;
import java.sql.BatchUpdateException;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.util.HashSet;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.Quote;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CSVPrinter_ESTest extends CSVPrinter_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVPrinter cSVPrinter0 = new CSVPrinter(charArrayWriter0, cSVFormat0);
      CharArrayWriter charArrayWriter1 = (CharArrayWriter)cSVPrinter0.getOut();
      assertEquals("", charArrayWriter1.toString());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CharBuffer charBuffer0 = CharBuffer.allocate(24);
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVPrinter cSVPrinter0 = new CSVPrinter(charBuffer0, cSVFormat0);
      cSVPrinter0.close();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVPrinter cSVPrinter0 = new CSVPrinter(charArrayWriter0, cSVFormat0);
      cSVPrinter0.close();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CharBuffer charBuffer0 = CharBuffer.allocate(24);
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVPrinter cSVPrinter0 = new CSVPrinter(charBuffer0, cSVFormat0);
      cSVPrinter0.flush();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVPrinter cSVPrinter0 = new CSVPrinter(charArrayWriter0, cSVFormat0);
      cSVPrinter0.flush();
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      BatchUpdateException batchUpdateException0 = new BatchUpdateException();
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      CSVPrinter cSVPrinter0 = cSVFormat0.print(charArrayWriter0);
      Object[] objectArray0 = new Object[1];
      objectArray0[0] = (Object) batchUpdateException0;
      cSVPrinter0.printRecords(objectArray0);
      assertEquals(42, charArrayWriter0.size());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      Object[] objectArray0 = new Object[14];
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("Kle\"_&hA5zXS\"_~[Hu");
      String string0 = cSVFormat1.format(objectArray0);
      assertEquals("\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\"", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat('4');
      Object[] objectArray0 = new Object[6];
      String string0 = cSVFormat0.format(objectArray0);
      assertEquals("44444", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      Object[] objectArray0 = new Object[6];
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("\r\n");
      String string0 = cSVFormat1.format(objectArray0);
      assertEquals("\\r\\n\t\\r\\n\t\\r\\n\t\\r\\n\t\\r\\n\t\\r\\n", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      Object[] objectArray0 = new Object[14];
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("Kle\"_&hA5\\zXS\"_~[Hu\tKle\"_&hA5\\zXS\"_~[Hu\tKle\"_&hA5\\zXS\"_~[Hu\tKle\"_&hA5\\zXS\"_~[Hu\tKle\"_&hA5\\zXS\"_~[Hu\tKle\"_&hA5\\zXS\"_~[Hu\tKle\"_&hA5\\zXS\"_~[Hu\tKle\"_&hA5\\zXS\"_~[Hu\tKle\"_&hA5\\zXS\"_~[Hu\tKle\"_&hA5\\zXS\"_~[Hu\tKle\"_&hA5\\zXS\"_~[Hu\tKle\"_&hA5\\zXS\"_~[Hu\tKle\"_&hA5\\zXS\"_~[Hu\tKle\"_&hA5\\zXS\"_~[Hu");
      String string0 = cSVFormat1.format(objectArray0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      Quote quote0 = Quote.NONE;
      CSVFormat cSVFormat1 = cSVFormat0.withQuotePolicy(quote0);
      Character character0 = new Character('9');
      CSVFormat cSVFormat2 = cSVFormat1.withQuoteChar(character0);
      Object[] objectArray0 = new Object[6];
      CSVPrinter cSVPrinter0 = new CSVPrinter(charArrayWriter0, cSVFormat2);
      cSVPrinter0.printRecords(objectArray0);
      assertEquals("\n\n\n\n\n\n", charArrayWriter0.toString());
      assertEquals(6, charArrayWriter0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      Quote quote0 = Quote.ALL;
      CSVFormat cSVFormat1 = cSVFormat0.withQuotePolicy(quote0);
      BatchUpdateException batchUpdateException0 = new BatchUpdateException();
      CSVPrinter cSVPrinter0 = new CSVPrinter(charArrayWriter0, cSVFormat1);
      cSVPrinter0.printRecords((Iterable<?>) batchUpdateException0);
      assertEquals(44, charArrayWriter0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      Quote quote0 = Quote.NON_NUMERIC;
      CSVFormat cSVFormat1 = cSVFormat0.withQuotePolicy(quote0);
      StringWriter stringWriter0 = new StringWriter();
      CSVPrinter cSVPrinter0 = cSVFormat1.print(stringWriter0);
      Object[] objectArray0 = new Object[3];
      objectArray0[2] = (Object) (-3349);
      cSVPrinter0.printRecord(objectArray0);
      assertEquals(3, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      Object[] objectArray0 = new Object[14];
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\",\"Kle\"\"_&hA5zXS\"\"_~[Hu\"");
      String string0 = cSVFormat1.format(objectArray0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      Object[] objectArray0 = new Object[9];
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("629Vv^vt,tg");
      String string0 = cSVFormat1.format(objectArray0);
      assertEquals("\"629Vv^vt,tg\",\"629Vv^vt,tg\",\"629Vv^vt,tg\",\"629Vv^vt,tg\",\"629Vv^vt,tg\",\"629Vv^vt,tg\",\"629Vv^vt,tg\",\"629Vv^vt,tg\",\"629Vv^vt,tg\"", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      Object[] objectArray0 = new Object[8];
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("<[|oB'2t");
      String string0 = cSVFormat1.format(objectArray0);
      assertEquals("\"<[|oB'2t\",<[|oB'2t,<[|oB'2t,<[|oB'2t,<[|oB'2t,<[|oB'2t,<[|oB'2t,<[|oB'2t", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      Object[] objectArray0 = new Object[3];
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("]SmSc/:Ig");
      String string0 = cSVFormat1.format(objectArray0);
      assertEquals("\"]SmSc/:Ig\",]SmSc/:Ig,]SmSc/:Ig", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      Object[] objectArray0 = new Object[14];
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("|4");
      String string0 = cSVFormat1.format(objectArray0);
      assertEquals("\"|4\"\t|4\t|4\t|4\t|4\t|4\t|4\t|4\t|4\t|4\t|4\t|4\t|4\t|4", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      Object[] objectArray0 = new Object[2];
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("Unexpected Quote value: ");
      String string0 = cSVFormat1.format(objectArray0);
      assertEquals("\"Unexpected Quote value: \",\"Unexpected Quote value: \"", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      StringWriter stringWriter0 = new StringWriter();
      CSVPrinter cSVPrinter0 = cSVFormat0.print(stringWriter0);
      cSVPrinter0.printComment("");
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentStart('C');
      StringWriter stringWriter0 = new StringWriter();
      CSVPrinter cSVPrinter0 = cSVFormat1.print(stringWriter0);
      cSVPrinter0.print(cSVFormat1);
      cSVPrinter0.printComment("\r\n");
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentStart('^');
      StringWriter stringWriter0 = new StringWriter();
      CSVPrinter cSVPrinter0 = cSVFormat1.print(stringWriter0);
      cSVPrinter0.printComment("\rl_:\n");
      assertEquals("^ \r\n^ l_:\r\n^ \r\n", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentStart('9');
      StringWriter stringWriter0 = new StringWriter();
      CSVPrinter cSVPrinter0 = cSVFormat1.print(stringWriter0);
      cSVPrinter0.printComment("\r");
      assertEquals("9 \r\n9 \r\n", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVPrinter cSVPrinter0 = cSVFormat0.print(charArrayWriter0);
      HashSet<ByteArrayOutputStream> hashSet0 = new HashSet<ByteArrayOutputStream>();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      hashSet0.add(byteArrayOutputStream0);
      cSVPrinter0.printRecords((Iterable<?>) hashSet0);
      assertEquals(42, charArrayWriter0.size());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      StringWriter stringWriter0 = new StringWriter();
      CSVPrinter cSVPrinter0 = cSVFormat0.print(stringWriter0);
      ResultSetMetaData resultSetMetaData0 = mock(ResultSetMetaData.class, new ViolatedAssumptionAnswer());
      doReturn(20).when(resultSetMetaData0).getColumnCount();
      ResultSet resultSet0 = mock(ResultSet.class, new ViolatedAssumptionAnswer());
      doReturn(resultSetMetaData0).when(resultSet0).getMetaData();
      doReturn((String) null, (String) null, (String) null, (String) null, (String) null).when(resultSet0).getString(anyInt());
      doReturn(true, false).when(resultSet0).next();
      cSVPrinter0.printRecords(resultSet0);
      assertEquals("\"\",,,,,,,,,,,,,,,,,,,\r\n", stringWriter0.toString());
  }
}