/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:33:57 GMT 2023
 */

package com.google.gson;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.gson.TypeInfoFactory;
import java.lang.reflect.Field;
import java.lang.reflect.Type;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeInfoFactory_ESTest extends TypeInfoFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      // Undeclared exception!
      try { 
        TypeInfoFactory.getTypeInfoForArray((Type) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // condition failed: false
         //
         verifyException("com.google.gson.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Type[] typeArray0 = new Type[7];
      // Undeclared exception!
      try { 
        TypeInfoFactory.getTypeInfoForField((Field) null, typeArray0[0]);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Type 'null' is not a Class, ParameterizedType, or GenericArrayType. Can't extract class.
         //
         verifyException("com.google.gson.TypeUtils", e);
      }
  }
}