/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:04:00 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.cli.PatternOptionBuilder;
import org.apache.commons.cli.TypeHandler;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeHandler_ESTest extends TypeHandler_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      // Undeclared exception!
      try { 
        TypeHandler.createValue("4", (Object) "4");
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // java.lang.String cannot be cast to java.lang.Class
         //
         verifyException("org.apache.commons.cli.TypeHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Object object0 = TypeHandler.createObject("VHA5O ");
      Object object1 = TypeHandler.createValue("VHA5O ", ((PatternOptionBuilder) object0).CLASS_VALUE);
      assertNull(object1);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Object object0 = TypeHandler.createObject("                                                                ");
      Object object1 = TypeHandler.createValue("                                                                ", ((PatternOptionBuilder) object0).NUMBER_VALUE);
      assertNull(object1);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Object object0 = TypeHandler.createObject("org.apache.commons.cli.PatternOptionBuilder");
      Object object1 = TypeHandler.createValue("org.apache.commons.cli.PatternOptionBuilder", ((PatternOptionBuilder) object0).FILES_VALUE);
      assertNull(object1);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TypeHandler typeHandler0 = new TypeHandler();
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Object object0 = TypeHandler.createObject("VHA5O");
      MockFile mockFile0 = (MockFile)TypeHandler.createValue("VHA5O", ((PatternOptionBuilder) object0).EXISTING_FILE_VALUE);
      assertFalse(mockFile0.isDirectory());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<String> class0 = String.class;
      Object object0 = TypeHandler.createValue("", class0);
      assertEquals("", object0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Object object0 = TypeHandler.createValue("-0x", class0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Object object0 = TypeHandler.createObject("");
      Object object1 = TypeHandler.createValue("", ((PatternOptionBuilder) object0).DATE_VALUE);
      assertNull(object1);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Object object0 = TypeHandler.createObject("0x");
      MockFile mockFile0 = (MockFile)TypeHandler.createValue("0x", ((PatternOptionBuilder) object0).FILE_VALUE);
      assertEquals(0L, mockFile0.getUsableSpace());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      Object object0 = TypeHandler.createValue("", class0);
      assertNull(object0);
  }
}
