/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:55:45 GMT 2023
 */

package org.apache.commons.lang3.text.translate;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.StringWriter;
import org.apache.commons.lang3.text.translate.NumericEntityUnescaper;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NumericEntityUnescaper_ESTest extends NumericEntityUnescaper_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      NumericEntityUnescaper numericEntityUnescaper0 = new NumericEntityUnescaper();
      String string0 = numericEntityUnescaper0.translate((CharSequence) "&");
      assertEquals("&", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      NumericEntityUnescaper numericEntityUnescaper0 = new NumericEntityUnescaper();
      String string0 = numericEntityUnescaper0.translate((CharSequence) "!Bf$zZuO9-2,&2DORn");
      assertEquals("!Bf$zZuO9-2,&2DORn", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      NumericEntityUnescaper numericEntityUnescaper0 = new NumericEntityUnescaper();
      StringWriter stringWriter0 = new StringWriter();
      StringWriter stringWriter1 = stringWriter0.append((CharSequence) "Pz&#");
      StringWriter stringWriter2 = stringWriter1.append('x');
      StringBuffer stringBuffer0 = stringWriter2.getBuffer();
      // Undeclared exception!
      try { 
        numericEntityUnescaper0.translate((CharSequence) stringBuffer0);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      NumericEntityUnescaper numericEntityUnescaper0 = new NumericEntityUnescaper();
      StringWriter stringWriter0 = new StringWriter();
      StringWriter stringWriter1 = stringWriter0.append((CharSequence) "&#");
      char[] charArray0 = new char[5];
      charArray0[0] = '3';
      charArray0[1] = ';';
      stringWriter1.write(charArray0);
      StringBuffer stringBuffer0 = stringWriter1.getBuffer();
      String string0 = numericEntityUnescaper0.translate((CharSequence) stringBuffer0);
      assertEquals("\u0003\u0000\u0000\u0000", string0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      NumericEntityUnescaper numericEntityUnescaper0 = new NumericEntityUnescaper();
      StringWriter stringWriter0 = new StringWriter();
      StringWriter stringWriter1 = stringWriter0.append((CharSequence) "&#");
      char[] charArray0 = new char[10];
      charArray0[0] = '3';
      charArray0[1] = '3';
      charArray0[2] = '3';
      charArray0[3] = '3';
      charArray0[4] = '3';
      stringWriter0.append('X');
      charArray0[5] = ';';
      stringWriter0.write(charArray0);
      StringBuffer stringBuffer0 = stringWriter1.getBuffer();
      String string0 = numericEntityUnescaper0.translate((CharSequence) stringBuffer0);
      assertEquals("\uD88C\uDF33\u0000\u0000\u0000\u0000", string0);
  }
}
