/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:15:49 GMT 2023
 */

package com.google.gson;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.gson.GenericArrayTypeImpl;
import com.google.gson.ParameterizedTypeImpl;
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
      Type[] typeArray0 = new Type[18];
      // Undeclared exception!
      try { 
        TypeInfoFactory.getTypeInfoForArray(typeArray0[0]);
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
      GenericArrayTypeImpl genericArrayTypeImpl0 = new GenericArrayTypeImpl((Type) null);
      Type[] typeArray0 = new Type[1];
      ParameterizedTypeImpl parameterizedTypeImpl0 = new ParameterizedTypeImpl(genericArrayTypeImpl0, typeArray0, genericArrayTypeImpl0);
      // Undeclared exception!
      try { 
        TypeInfoFactory.getTypeInfoForField((Field) null, parameterizedTypeImpl0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Type 'null' is not a Class, ParameterizedType, or GenericArrayType. Can't extract class.
         //
         verifyException("com.google.gson.TypeUtils", e);
      }
  }
}
